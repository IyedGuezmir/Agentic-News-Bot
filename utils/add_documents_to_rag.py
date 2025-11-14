# utils/add_documents.py
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def add_documents(csv_path, doc_limit=None):
    # Load CSV
    loader = CSVLoader(file_path=csv_path, encoding="utf-8")
    documents = loader.load()
    if doc_limit:
        documents = documents[:doc_limit]
    print(f"[ADD DOCS] Loaded {len(documents)} documents from CSV.")

    # Convert to graph documents
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    transformer = LLMGraphTransformer(llm)
    graph_docs = transformer.convert_to_graph_documents(documents)
    print("[ADD DOCS] Converted documents to graph format.")

    # Connect to Neo4j and add documents
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    graph.add_graph_documents(graph_docs, include_source=True, baseEntityLabel=True)
    print("[ADD DOCS] Documents added to Neo4j.")
