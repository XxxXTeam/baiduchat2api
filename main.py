import os
import time
import argparse
import json
from typing import Optional, Dict, Any

from flask import Flask, request, jsonify, Response
from baidu_chat import BaiduChatClient, _log
from tool_calling import messages_to_prompt, parse_tool_calls


# ------------------------------------------------------------------
# Config loader
# ------------------------------------------------------------------
def load_config(path: str = "config.toml") -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        pass
    try:
        import tomli
        with open(path, "rb") as f:
            return tomli.load(f)
    except Exception:
        pass
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}


# ------------------------------------------------------------------
# Flask App
# ------------------------------------------------------------------
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = True
client: Optional[BaiduChatClient] = None


MODEL_LIST = [
    {"id": "baidu-ernie-4.5", "object": "model", "created": int(time.time()), "owned_by": "baidu"},
    {"id": "baidu-ernie-4.5-think", "object": "model", "created": int(time.time()), "owned_by": "baidu"},
    {"id": "baidu-deepseek-r1", "object": "model", "created": int(time.time()), "owned_by": "baidu"},
    {"id": "baidu-deepseek-r1-think", "object": "model", "created": int(time.time()), "owned_by": "baidu"},
    {"id": "baidu-deepseek-v4-pro", "object": "model", "created": int(time.time()), "owned_by": "baidu"},
    {"id": "baidu-deepseek-v4-pro-think", "object": "model", "created": int(time.time()), "owned_by": "baidu"},
]

MODEL_MAP = {
    "baidu-ernie-4.5": "ernie-4.5",
    "baidu-ernie-4.5-think": "ernie-4.5-think",
    "baidu-wenxin": "ernie-4.5",
    "baidu-wenxin-think": "ernie-4.5-think",
    "baidu-smart": "ernie-4.5",
    "baidu-smart-think": "ernie-4.5-think",
    "baidu-deepseek-r1": "deepseek-r1",
    "baidu-deepseek-r1-think": "deepseek-r1-think",
    "baidu-deepseek": "deepseek-r1",
    "baidu-deepseek-think": "deepseek-r1-think",
    "baidu-deepseek-v4-pro": "deepseek-v4-pro",
    "baidu-deepseek-v4-pro-think": "deepseek-v4-pro-think",
    "baidu-dsv4pro": "deepseek-v4-pro",
    "baidu-dsv4pro-think": "deepseek-v4-pro-think",
    "baidu-ds-v4": "deepseek-v4-pro",
    "baidu-ds-v4-think": "deepseek-v4-pro-think",
    "gpt-3.5-turbo": "ernie-4.5",
    "gpt-4": "deepseek-r1",
    "gpt-4-turbo": "deepseek-v4-pro",
}


def _error(message: str, status: int = 400, err_type: str = "invalid_request"):
    return jsonify({"error": {"message": message, "type": err_type}}), status


def _resolve_server_config(config: Dict[str, Any], host: str, port: int) -> tuple[str, int]:
    server_cfg = config.get("server", {})
    if isinstance(server_cfg, dict):
        host = server_cfg.get("host", host)
        port = int(server_cfg.get("port", port))
    return host, port


def _resolve_client_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cookies_cfg = config.get("cookies", {})
    cookies = cookies_cfg.get("value", "") if isinstance(cookies_cfg, dict) else str(cookies_cfg)
    headers_cfg = config.get("headers", {})
    persistence_cfg = config.get("cookie_persistence", {})

    return {
        "cookies": cookies or None,
        "user_agent": headers_cfg.get("user_agent") if isinstance(headers_cfg, dict) else None,
        "cookie_file": (
            persistence_cfg.get("cookie_file")
            if isinstance(persistence_cfg, dict)
            else config.get("cookie_file")
        ) or "cookies.json",
        "auto_save_cookies": (
            persistence_cfg.get("auto_save_cookies")
            if isinstance(persistence_cfg, dict)
            else config.get("auto_save_cookies", True)
        ),
    }


@app.route("/v1/models", methods=["GET"])
def list_models():
    _log("INFO", f"GET /v1/models  from {request.remote_addr}")
    return jsonify({"object": "list", "data": MODEL_LIST})


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    req = request.get_json(force=True, silent=True) or {}
    if not req:
        return _error("Invalid JSON body")

    model = req.get("model", "baidu-ernie-4.5")
    messages = req.get("messages", [])
    tools = req.get("tools") or []
    tool_choice = req.get("tool_choice")
    stream = req.get("stream", False)

    baidu_model = MODEL_MAP.get(model, "ernie-4.5")
    deep_search = bool(req.get("deep_search", False))
    query = messages_to_prompt(messages, tools if isinstance(tools, list) else [], tool_choice)

    if not query:
        return _error("No user message found")

    _log("INFO", f"POST /v1/chat/completions  model={model}  stream={stream}  tools={len(tools) if isinstance(tools, list) else 0}  query_len={len(query)}")

    has_tools = isinstance(tools, list) and bool(tools)
    if stream:
        return _handle_stream(query, baidu_model, deep_search, model, has_tools)
    else:
        return _handle_sync(query, baidu_model, deep_search, model)


def _handle_stream(query: str, baidu_model: str, deep_search: bool, display_model: str, has_tools: bool = False):
    if not client:
        return _error("Client not initialized", 500, "internal_error")

    def _sse(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def generate():
        yield _sse({
            "id": "chatcmpl-baidu",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": display_model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        })

        try:
            content_parts = []
            reasoning_parts = []
            for chunk in client.chat_to_openai_chunks(query, model=baidu_model, deep_search=deep_search):
                content = chunk.get("content")
                if not content:
                    continue
                if chunk["type"] == "content":
                    content_parts.append(content)
                    if not has_tools:
                        yield _sse({
                            "id": "chatcmpl-baidu",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": display_model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": None,
                            }],
                        })
                elif chunk["type"] == "reasoning_content":
                    reasoning_parts.append(content)
                    yield _sse({
                        "id": "chatcmpl-baidu",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": display_model,
                        "choices": [{
                            "index": 0,
                            "delta": {"reasoning_content": content},
                            "finish_reason": None,
                        }],
                    })

            parsed_content, tool_calls = parse_tool_calls("".join(content_parts))
            if tool_calls:
                if parsed_content:
                    yield _sse({
                        "id": "chatcmpl-baidu",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": display_model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": parsed_content},
                            "finish_reason": None,
                        }],
                    })
                for idx, tool_call in enumerate(tool_calls):
                    yield _sse({
                        "id": "chatcmpl-baidu",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": display_model,
                        "choices": [{
                            "index": 0,
                            "delta": {"tool_calls": [{
                                "index": idx,
                                "id": tool_call["id"],
                                "type": "function",
                                "function": tool_call["function"],
                            }]},
                            "finish_reason": None,
                        }],
                    })
                yield _sse({
                    "id": "chatcmpl-baidu",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": display_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                })
                yield "data: [DONE]\n\n"
                return
            if has_tools and parsed_content:
                yield _sse({
                    "id": "chatcmpl-baidu",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": display_model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": parsed_content},
                        "finish_reason": None,
                    }],
                })
        except Exception as e:
            _log("ERROR", f"Stream error: {e}")
            yield _sse({"error": str(e)})
            return

        yield _sse({
            "id": "chatcmpl-baidu",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": display_model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")


def _handle_sync(query: str, baidu_model: str, deep_search: bool, display_model: str):
    if not client:
        return _error("Client not initialized", 500, "internal_error")

    try:
        result = client.chat_to_openai_sync(query, model=baidu_model, deep_search=deep_search)
        content, tool_calls = parse_tool_calls(result.get("content", ""))
        message = {
            "role": "assistant",
            "content": content,
        }
        if result.get("reasoning_content"):
            message["reasoning_content"] = result["reasoning_content"]
        finish_reason = "stop"
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        return jsonify({
            "id": "chatcmpl-baidu",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": display_model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
        })
    except Exception as e:
        _log("ERROR", f"Sync error: {e}")
        return jsonify({"error": {"message": str(e), "type": "internal_error"}}), 500


# ------------------------------------------------------------------
# Startup
# ------------------------------------------------------------------
def run_server(host: str = "0.0.0.0", port: int = 8000, config: Optional[Dict[str, Any]] = None):
    global client
    config = config or {}

    host, port = _resolve_server_config(config, host, port)
    client_cfg = _resolve_client_config(config)

    client = BaiduChatClient(
        cookies=client_cfg["cookies"],
        user_agent=client_cfg["user_agent"],
        cookie_file=client_cfg["cookie_file"],
        auto_save_cookies=bool(client_cfg["auto_save_cookies"]),
    )

    _log("INFO", f"Flask server starting at http://{host}:{port}")
    cookie_mode = "user-provided" if client_cfg["cookies"] else f"auto-fetch + file={client_cfg['cookie_file']}"
    _log("INFO", f"Cookie mode: {cookie_mode}")
    _log("INFO", "Models: baidu-ernie-4.5[-think], baidu-deepseek-r1[-think], baidu-deepseek-v4-pro[-think]")
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baidu Chat OpenAI-compatible API server (Flask)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--config", default="config.toml", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    host, port = _resolve_server_config(cfg, args.host, args.port)
    run_server(host=host, port=port, config=cfg)
