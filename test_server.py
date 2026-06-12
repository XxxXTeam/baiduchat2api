import json
import os
import time
import sys
import socket
import threading
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import app, client as main_client
from baidu_chat import BaiduChatClient


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_flask_server():
    """Test the Flask-based OpenAI-compatible server with a mock Baidu client."""
    port = find_free_port()

    # Create a mock client
    mock_client = BaiduChatClient(cookies=None, cookie_file=None, auto_save_cookies=False)
    mock_client._token = "test_token"
    mock_client._lid = "test_lid"
    mock_client._ori_lid = "test_lid"

    def fake_stream(query, model="smart", deep_search=False, internet_search=False):
        yield {"type": "text", "content": "Hello"}
        yield {"type": "thinking", "content": "Let me think..."}
        yield {"type": "text", "content": " world!"}
        yield {"type": "done", "content": ""}

    mock_client.chat_stream_text = fake_stream

    # Patch the global client in main module
    import main as main_mod
    main_mod.client = mock_client

    # Configure Flask test client
    app.config["TESTING"] = True
    with app.test_client() as tc:
        results = []

        # Test 1: GET /v1/models
        r = tc.get("/v1/models")
        data = r.get_json()
        results.append({
            "test": "models",
            "status": r.status_code,
            "models": [m["id"] for m in data.get("data", [])]
        })

        # Test 2: POST sync
        r = tc.post("/v1/chat/completions",
                    json={"model": "baidu-smart", "messages": [{"role": "user", "content": "hello"}], "stream": False})
        data = r.get_json()
        msg = data.get("choices", [{}])[0].get("message", {})
        results.append({
            "test": "chat_sync",
            "status": r.status_code,
            "content": msg.get("content", "")[:50],
            "has_reasoning": msg.get("reasoning_content") is not None,
        })

        # Test 3: POST stream
        r = tc.post("/v1/chat/completions",
                    json={"model": "baidu-smart", "messages": [{"role": "user", "content": "hello"}], "stream": True})
        raw = r.data.decode("utf-8")
        lines = [ln for ln in raw.strip().split("\n") if ln.startswith("data: ")]
        chunks = []
        for ln in lines:
            payload = ln[6:]
            if payload == "[DONE]":
                chunks.append("[DONE]")
                continue
            try:
                d = json.loads(payload)
                delta = d.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    chunks.append(delta["content"])
                elif delta.get("role"):
                    chunks.append(f"[role:{delta['role']}]")
                elif d.get("choices", [{}])[0].get("finish_reason"):
                    chunks.append("[finish]")
            except json.JSONDecodeError:
                chunks.append(f"[raw:{payload}]")
        results.append({
            "test": "chat_stream",
            "status": r.status_code,
            "chunks": chunks,
            "full_text": "".join([c for c in chunks if not c.startswith("[")]),
        })

        # Test 4: DeepSeek model
        r = tc.post("/v1/chat/completions",
                    json={"model": "baidu-deepseek", "messages": [{"role": "user", "content": "hello"}], "stream": False})
        data = r.get_json()
        msg = data.get("choices", [{}])[0].get("message", {})
        results.append({
            "test": "deepseek_sync",
            "status": r.status_code,
            "has_reasoning": msg.get("reasoning_content") is not None,
            "content": msg.get("content", "")[:50],
        })

    return results


if __name__ == "__main__":
    results = test_flask_server()
    print(json.dumps(results, ensure_ascii=False, indent=2))
