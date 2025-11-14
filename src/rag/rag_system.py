# utils/rag_system.py
from langchain_openai import ChatOpenAI
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
import os

class GraphRAGSystem:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            validate_cypher=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True
        )
    def retrieve(self, question: str):
        """
        Run NL -> Cypher -> Neo4j and return the raw graph/context,
        """
        out = self.chain.invoke({"query": question})
        # `out` typically has: {"result": "...", "intermediate_steps": ...}
        # We ignore `result` and only expose the context.
        return out.get("intermediate_steps", out)

    def query(self, query_text):
        return self.chain.invoke({"query": query_text})
