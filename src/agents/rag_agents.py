import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from src.rag.hybrid_rag_system import HybridRAGSystem
from src.rag.graph_rag_system import GraphRAGSystem


# Instantiate RAG systems once per process
_hybrid_rag = HybridRAGSystem()
_graph_rag = GraphRAGSystem()


@tool
def hybrid_rag_qa(question: str) -> dict:
    """
    Answer questions using the hybrid RAG pipeline over the news CSV
    (dense + BM25 + reranker). Returns answer and raw contexts.
    """
    answer = _hybrid_rag.query(question)
    contexts = _hybrid_rag.retrieve(question)
    return {
        "type": "hybrid_rag",
        "question": question,
        "answer": answer,
        "contexts": [d.page_content for d in contexts],
    }


@tool
def graph_rag_qa(question: str) -> dict:
    """
    Answer questions using the Neo4j Graph RAG pipeline.
    Returns answer and intermediate graph steps.
    """
    result = _graph_rag.query(question)
    # GraphRAGSystem.query currently returns whatever GraphCypherQAChain returns,
    # typically a dict with "result" and "intermediate_steps".
    return {
        "type": "graph_rag",
        "question": question,
        "answer": result.get("result", result),
        "raw": result,
    }


def create_hybrid_rag_agent(llm: ChatOpenAI):
    """
    Agent that ALWAYS uses hybrid_rag_qa for questions it receives.
    """
    return create_react_agent(
        llm,
        tools=[hybrid_rag_qa],
        prompt=(
            "You are a news QA agent over a CSV corpus of news articles.\n\n"
            "INSTRUCTIONS:\n"
            "- You MUST answer by calling the tool hybrid_rag_qa exactly once.\n"
            "- Do not do free-form reasoning; just delegate to the tool.\n"
            "- Return only the tool JSON result, no extra prose."
        ),
        name="hybrid_rag",
    )


def create_graph_rag_agent(llm: ChatOpenAI):
    """
    Agent that ALWAYS uses graph_rag_qa for questions it receives.
    """
    return create_react_agent(
        llm,
        tools=[graph_rag_qa],
        prompt=(
            "You are a graph QA agent over a Neo4j news knowledge graph.\n\n"
            "INSTRUCTIONS:\n"
            "- You MUST answer by calling graph_rag_qa exactly once.\n"
            "- Use it for relationship / multi-hop / entity-centric queries.\n"
            "- Return only the tool JSON result, no extra prose."
        ),
        name="graph_rag",
    )