from pydantic import BaseModel, Field
from agents import Agent, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)
# Use Claude as the model, through Anthropic's OpenAI-compatible endpoint.
# claude-haiku-5-5 is the cheap default; set DEFAULT_MODEL_NAME in .env to use another Claude model.
CLAUDE_MODEL = os.getenv("DEFAULT_MODEL_NAME", "claude-haiku-5-5")
claude_client = AsyncOpenAI(base_url="https://api.anthropic.com/v1/", api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL_NAME = OpenAIChatCompletionsModel(model=CLAUDE_MODEL, openai_client=claude_client)

INSTRUCTIONS = """
You are a senior researcher tasked with writing a cohesive report for a research query.
You will be provided with the original query, and some research.
Generate a comprehensive report based on the research and the query.
The final output should be in markdown format, and it should be lengthy and detailed. Aim 
for 5-10 pages of content, at least 1000 words.
"""


class ReportData(BaseModel):
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")
    markdown_report: str = Field(description="The final report")
    follow_up_questions: list[str] = Field(description="Suggested topics to research further")


writer_agent = Agent(name="Writer Agent", instructions=INSTRUCTIONS, model=MODEL_NAME, output_type=ReportData)
