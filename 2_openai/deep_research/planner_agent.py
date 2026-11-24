from pydantic import BaseModel, Field
from agents import Agent, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# Use Claude as the model, through Anthropic's OpenAI-compatible endpoint.
# claude-haiku-5-5 is the cheap default; set DEFAULT_MODEL_NAME in .env to use another Claude model.
CLAUDE_MODEL = os.getenv("DEFAULT_MODEL_NAME", "claude-haiku-5-5")
claude_client = AsyncOpenAI(base_url="https://api.anthropic.com/v1/", api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL_NAME = OpenAIChatCompletionsModel(model=CLAUDE_MODEL, openai_client=claude_client)
HOW_MANY_SEARCHES = int(os.getenv("HOW_MANY_SEARCHES", 5))


INSTRUCTIONS = f"""
You are a research assistant. Given a user query, come up with a set of web searches
to perform to best answer the query. Output {HOW_MANY_SEARCHES} terms to query for.
"""

class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")
    
planner_agent = Agent(name="Planner Agent", instructions=INSTRUCTIONS, model=MODEL_NAME, output_type=WebSearchPlan)