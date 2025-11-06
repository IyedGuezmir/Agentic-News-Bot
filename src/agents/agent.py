import os
import sys

# Ensure project root is on sys.path for imports like `utils.*`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_openai import ChatOpenAI
try:
    # Prefer user's import style if available
    from langgraph_supervisor import create_supervisor  # type: ignore
except ImportError:  # fallback for environments without `langgraph_supervisor`
    from langgraph.prebuilt import create_supervisor  # type: ignore
from langgraph.checkpoint.memory import InMemorySaver

# Import per-agent factories
from src.agents.content_creator_agent import create_content_creator_agent
from src.agents.analyst_agent import create_analyst_agent
from src.agents.detector_agent import create_detector_agent

# Shared LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Instantiate agents
content_creator_agent = create_content_creator_agent(llm)
analysis_agent = create_analyst_agent(llm)
detector_agent = create_detector_agent(llm)

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
