import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class GraphRAGSystem:
    """
    Lightweight Graph RAG over Neo4j, without GraphCypherQAChain.

    - retrieve(question): generate a Cypher query, run it, and return rows
    - query(question): use retrieved rows + LLM to answer the question
    """

    def __init__(self) -> None:
        # Neo4j connection
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
        )

        # LLM
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Prompt to generate Cypher
        self.cypher_prompt = ChatPromptTemplate.from_template(
            """You are an expert at writing Cypher for a Neo4j graph database.
Given the user's natural language question, write ONE Cypher query that can be
run against the graph to answer it.

Return ONLY the Cypher query, no explanation, no backticks.

Question: {question}
"""
        )
        self.cypher_parser = StrOutputParser()

        # Prompt to answer using graph rows
        self.answer_prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant answering questions using graph data.

You are given:
- The original question.
- Rows returned from a Cypher query over a Neo4j news graph.

Use ONLY this information to answer the question. If the graph data is not
sufficient, say you don't know.

Question:
{question}

Graph rows (as JSON-like text):
{rows}

Answer clearly and concisely."""
        )
        self.answer_parser = StrOutputParser()

    def _generate_cypher(self, question: str) -> str:
        """Use the LLM to propose a Cypher query for the question."""
        chain = self.cypher_prompt | self.llm | self.cypher_parser
        cypher = chain.invoke({"question": question}).strip()
        # Optional: sanity check / basic guardrails could go here
        return cypher

    def retrieve(self, question: str):
        """
        Generate Cypher from the question, execute it, and return the raw rows.

        Returns a dict with keys:
        - cypher: the generated Cypher query string
        - rows: list[dict] from Neo4jGraph.query(...)
        """
        cypher = self._generate_cypher(question)
        rows = self.graph.query(cypher)
        return {"cypher": cypher, "rows": rows}

    def query(self, question: str) -> str:
        """
        Run full Graph RAG: generate Cypher, run it, then answer with the LLM.

        Returns the final answer text.
        """
        ctx = self.retrieve(question)
        rows_text = str(ctx["rows"])

        chain = self.answer_prompt | self.llm | self.answer_parser
        answer = chain.invoke({"question": question, "rows": rows_text})
        return answer