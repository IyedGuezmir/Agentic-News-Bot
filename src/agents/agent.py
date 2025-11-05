import os
import sys

# Ensure project root is on sys.path for imports like `utils.*`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
try:
    # Prefer user's import style if available
    from langgraph_supervisor import create_supervisor  # type: ignore
except ImportError:  # fallback for environments without `langgraph_supervisor`
    from langgraph.prebuilt import create_supervisor  # type: ignore
from langgraph.checkpoint.memory import InMemorySaver

# Import tools
from utils.tools import (
    generate_news_article,
    summarize_text,
    analyze_sentiment,
    ml_predict_news,
    web_search_verify,
)

# Shared LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Content Creator agent
content_creator_agent = create_react_agent(
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

# Analysis agent (summarization + sentiment)
analysis_agent = create_react_agent(
    llm,
    tools=[summarize_text, analyze_sentiment],
    prompt=(
        "You are an analysis agent for summarization and sentiment.\n\n"
        "INSTRUCTIONS:\n"
        "- You MUST use tools, not free-form replies.\n"
        "- Given an article {title, text}, call summarize_text once, then analyze_sentiment once.\n"
        "- Return a compact JSON with keys: summary, sentiment (the tool outputs). No extra prose."
    ),
    name="analyst",
)

# News detection agent (fake news detection)
detector_agent = create_react_agent(
    llm,
    tools=[ml_predict_news, web_search_verify],
    prompt=(
        "You are a verification agent that detects fake news.\n\n"
        "INSTRUCTIONS:\n"
        "- Operate on ONE article ({title, subject, date, text}).\n"
        "- Call ml_predict_news exactly once. If confidence < 85%, optionally call web_search_verify once.\n"
        "- Return a JSON: { prediction, confidence, method, optionally: verified, source_url, reasoning }.\n"
        "- No extra prose."
    ),
    name="detector",
)

# Supervisor graph
checkpointer = InMemorySaver()

def build_supervisor(thread_id: str = "1", user_id: str = "1"):
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    supervisor = create_supervisor(
        model=llm,
        agents=[content_creator_agent, analysis_agent, detector_agent],
prompt=(
            "You are a supervisor managing three agents:\n"
            "- content_creator: Generate exactly one article. If the user specifies a topic, pass it via 'topic'.\n"
            "- analyst: Summarize and run sentiment on that single article using tools only.\n"
            "- detector: Verify that same single article.\n"
            "Assign work to one agent at a time; do NOT call agents in parallel or repeat agents unless required.\n"
            "Do not attempt to generate multiple articles; stop after the first valid article is produced.\n"
            "Do not do any work yourself."
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile(checkpointer=checkpointer)

    return supervisor, config

# Default, ready-to-use compiled supervisor and a default config
supervisor, default_config = build_supervisor()
