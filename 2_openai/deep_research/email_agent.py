from agents import Agent, function_tool, ModelSettings, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from messenger import send_email, push
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# Use Claude as the model, through Anthropic's OpenAI-compatible endpoint.
# claude-haiku-5-5 accepts tool_choice="required", which this agent needs (claude-opus-5-5 rejects it).
CLAUDE_MODEL = os.getenv("DEFAULT_MODEL_NAME", "claude-haiku-5-5")
claude_client = AsyncOpenAI(base_url="https://api.anthropic.com/v1/", api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL_NAME = OpenAIChatCompletionsModel(model=CLAUDE_MODEL, openai_client=claude_client)
USE_EMAIL = os.getenv("USE_EMAIL", "true").lower() == "true"

settings = ModelSettings(tool_choice="required")

@function_tool
def send_email_tool(subject: str, text_body: str, html_body: str) -> str:
    """
    Send out an email with the given subject and body
    
    Args:
        subject: The subject of the email
        text_body: The body of the email as plain text
        html_body: The HTML body of the email
    """
    if USE_EMAIL:
        send_email(subject, text_body, html_body)
    else:
        push(f"Subject: {subject}\n\n{text_body}")
    return "Email sent successfully"


INSTRUCTIONS = """
You are provided with a detailed report. Use your tool to send an email, converting the report into
a clean, well presented HTML email with an appropriate subject line.
"""

email_agent = Agent(name="Email Agent", instructions=INSTRUCTIONS, tools=[send_email_tool], model=MODEL_NAME, model_settings=settings)