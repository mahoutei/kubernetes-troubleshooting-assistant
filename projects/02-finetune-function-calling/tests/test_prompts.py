from src.data import Example
from src.prompts import build_messages, format_target, parse_tool_calls, render_example

TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]


def example():
    return Example("id", TOOLS, "do it", [{"name": "f", "arguments": {"x": 1}}], "test")


def test_build_messages_contains_tools_and_user():
    messages = build_messages(example())
    assert messages[0]["role"] == "system"
    assert "f" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "do it"}


def test_target_roundtrip_to_multiple_calls():
    calls = [{"name": "f", "arguments": {"x": 1}}, {"name": "g", "arguments": {"y": [1, 2]}}]
    text = format_target(calls)
    assert parse_tool_calls(text) == calls


def test_parse_ignores_malformed_call():
    text = '<tool_call>{oops}</tool_call><tool_call>{"name": "f", "arguments": {}}</tool_call>'
    assert parse_tool_calls(text) == [{"name": "f", "arguments": {}}]


def test_render_example_uses_chat_template_and_target():
    class FakeTokenizer:
        eos_token = "<eos>"
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is True
            return "\n".join(m["content"] for m in messages) + "\nASSISTANT:"

    text = render_example(example(), FakeTokenizer())
    assert text.endswith("<eos>")
    assert "<tool_call>" in text
    assert "ASSISTANT:" in text
