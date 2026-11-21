import anthropic
from context import TWIN_SYSTEM_PROMPT
from tools import tools, handle_tool_calls
from styles import CSS, JS, EXAMPLES
from dotenv import load_dotenv
import gradio as gr

load_dotenv(override=True)

MODEL_NAME = "claude-opus-5-5"

# Reads ANTHROPIC_API_KEY from the environment
claude = anthropic.Anthropic()


def get_text(response):
    """Claude returns a list of content blocks; join the text ones into one string."""
    return "".join(block.text for block in response.content if block.type == "text")


def clean_history(history):
    """Gradio's history entries carry extra fields that Claude's API rejects, so keep only role and text."""
    cleaned = []
    for h in history:
        content = h["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if isinstance(content, str) and content:
            cleaned.append({"role": h["role"], "content": content})
    return cleaned


def ask_claude(messages):
    # With Claude, the system prompt is passed separately from the messages
    return claude.messages.create(
        model=MODEL_NAME,
        max_tokens=4096,
        system=TWIN_SYSTEM_PROMPT,
        messages=messages,
        tools=tools,
    )


def chat(message, history):
    messages = clean_history(history) + [{"role": "user", "content": message}]
    response = ask_claude(messages)
    while response.stop_reason == "tool_use":
        tool_calls = [block for block in response.content if block.type == "tool_use"]
        results = handle_tool_calls(tool_calls)
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})
        response = ask_claude(messages)
    return get_text(response)


if __name__ == "__main__":
    gr.ChatInterface(
        chat,
        examples=EXAMPLES,
        title="Digital Twin",
        description="Talk to my AI twin about my career",
        chatbot=gr.Chatbot(show_label=False),
    ).launch(css=CSS, js=JS, theme=gr.themes.Base())
