from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, function_tool
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# Use Claude as the model, through Anthropic's OpenAI-compatible endpoint.
# claude-haiku-5-5 is cheap, which matters here because web searches read a lot of text.
# It also accepts tool_choice="required", which this agent needs (claude-opus-5-5 rejects it).
CLAUDE_MODEL = os.getenv("DEFAULT_MODEL_NAME", "claude-haiku-5-5")
claude_client = AsyncOpenAI(base_url="https://api.anthropic.com/v1/", api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL_NAME = OpenAIChatCompletionsModel(model=CLAUDE_MODEL, openai_client=claude_client)

INSTRUCTIONS = """
You are a research assistant. Given a search term, you search the web for that term and 
produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 words.
Capture the main points and be succinct. Reply only with the summary.
"""

# OpenAI's hosted WebSearchTool only works with OpenAI models.
# This tool does the same job with Claude: it asks Claude to search the web using Anthropic's web search.
anthropic_client = AsyncAnthropic()


@function_tool
async def web_search(query: str) -> str:
    """ Search the web for the given query and return what was found """
    response = await anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}],
        messages=[{"role": "user", "content": f"Search the web for: {query}\nReport the key facts you find, with sources."}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


settings = ModelSettings(tool_choice="required")
tools = [web_search]

search_agent = Agent(name="Search Agent", instructions=INSTRUCTIONS, tools=tools, model=MODEL_NAME, model_settings=settings)
