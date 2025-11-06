import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from utils.simulation_helpers import generate_single_news_structured_llm

# Content generation tool
@tool
def generate_news_article(topic: str = "", subject: str = "") -> dict:
    """
    Generate ONE news article (can be real or fake).
    Optional args:
    - topic: what the article should be about (e.g., "AI chips").
    - subject: category (e.g., "US_News", "worldnews").
    Returns JSON with title, text, subject, date, label.
    """
    try:
        t = topic.strip() or None
        s = subject.strip() or None
        news_item = generate_single_news_structured_llm(topic=t, subject=s)
        return {
            "success": True,
            "title": news_item.title,
            "text": news_item.text,
            "subject": news_item.subject,
            "date": news_item.date,
            "label": "real" if news_item.label == 1 else "fake"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Content Creator Agent
def create_content_creator_agent(llm: ChatOpenAI):
    return create_react_agent(
        llm,
        tools=[generate_news_article],
        prompt=(
            "You are a content creation agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Generate EXACTLY ONE article by calling generate_news_article ONCE.\n"
            "- If the user specifies a topic, pass it via the 'topic' argument; otherwise leave it empty.\n"
            "- Do NOT call the tool multiple times to search for better outputs. One call only.\n"
            "- After you're done, respond to the supervisor directly with ONLY the tool JSON."
        ),
        name="content_creator",
    )

# Export tools for standalone use
CONTENT_TOOLS = [generate_news_article]