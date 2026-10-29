from __future__ import annotations

import json

from src.data import Example, extract_tagged_json, iter_calls

SYSTEM_TEMPLATE = """You are a function-calling assistant.
Available tools:
<tools>
{tools}
</tools>

Respond ONLY with one or more tool calls in this exact format:
<tool_call>
{{"name": "tool_name", "arguments": {{...}}}}
</tool_call>
Do not add explanations or markdown."""


def build_messages(example: Example) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(tools=json.dumps(example.tools))},
        {"role": "user", "content": example.user},
    ]


def format_target(calls: list[dict]) -> str:
    return "\n".join(
        f"<tool_call>\n{json.dumps(call, ensure_ascii=False)}\n</tool_call>" for call in calls
    )


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for call in iter_calls(extract_tagged_json(text, "tool_call")):
        if isinstance(call.get("name"), str) and isinstance(call.get("arguments"), dict):
            calls.append({"name": call["name"], "arguments": call["arguments"]})
    return calls


def render_example(example: Example, tokenizer) -> str:
    prompt = tokenizer.apply_chat_template(
        build_messages(example), tokenize=False, add_generation_prompt=True
    )
    return prompt + format_target(example.calls) + tokenizer.eos_token
