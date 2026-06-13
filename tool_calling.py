import json
import re
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple


TOOL_SYSTEM_PROMPT = """You can call tools when they are useful.

When calling tools, do not answer in natural language. Output only XML in this exact format:
<tool_calls>
  <tool_call>
    <name>tool_name</name>
    <arguments>{{"arg":"value"}}</arguments>
  </tool_call>
</tool_calls>

Rules:
- Use only tool names listed below.
- arguments must be a valid JSON object.
- If a required tool is specified, call that tool.
- If no tool is needed, answer normally without XML.

Available tools:
{tools}

{tool_choice}
"""


def build_tool_system_prompt(tools: List[Dict[str, Any]], tool_choice: Any = None) -> str:
    summaries = []
    for tool in tools:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = function.get("name", "")
        if not name:
            continue
        description = function.get("description", "")
        parameters = function.get("parameters", {})
        summaries.append(json.dumps({
            "name": name,
            "description": description,
            "parameters": parameters,
        }, ensure_ascii=False))
    return TOOL_SYSTEM_PROMPT.format(
        tools="\n".join(summaries),
        tool_choice=_format_tool_choice(tool_choice),
    )


def messages_to_prompt(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    if not isinstance(messages, list):
        return ""

    parts = []
    if tools:
        parts.append(f"System: {build_tool_system_prompt(tools, tool_choice)}")

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = _content_to_text(msg.get("content", ""))
        if content:
            parts.append(f"{role.capitalize()}: {content}")

        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            parts.append(f"Assistant tool_calls: {json.dumps(tool_calls, ensure_ascii=False)}")

        if role == "tool":
            tool_name = msg.get("name") or msg.get("tool_call_id") or "tool"
            parts[-1] = f"Tool({tool_name}): {content}" if content else parts[-1]

    return "\n\n".join(parts).strip()


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))
        return "\n".join(text for text in texts if text)
    return ""


def _format_tool_choice(tool_choice: Any) -> str:
    if not tool_choice or tool_choice == "auto":
        return ""
    if tool_choice == "none":
        return "Required tool: none. Do not call tools."
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function", {})
        name = function.get("name", "") if isinstance(function, dict) else ""
        if name:
            return f"Required tool: {name}"
    return ""


def parse_tool_calls(content: str) -> Tuple[str, List[Dict[str, Any]]]:
    if not content:
        return content, []

    xml_blocks = re.findall(r"<tool_calls\b[^>]*>.*?</tool_calls>", content, flags=re.DOTALL)
    if not xml_blocks:
        return content, []

    tool_calls = []
    for block in xml_blocks:
        tool_calls.extend(_parse_tool_call_block(block))

    cleaned = content
    for block in xml_blocks:
        cleaned = cleaned.replace(block, "")
    return cleaned.strip(), tool_calls


def _parse_tool_call_block(block: str) -> List[Dict[str, Any]]:
    try:
        root = ET.fromstring(block)
    except ET.ParseError:
        return []

    parsed = []
    for node in root.findall("tool_call"):
        name = (node.findtext("name") or "").strip()
        raw_args = (node.findtext("arguments") or "{}").strip()
        if not name:
            continue
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        parsed.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    return parsed
