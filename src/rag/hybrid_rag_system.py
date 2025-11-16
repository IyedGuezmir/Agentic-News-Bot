# src/rag/hybrid_rag_system.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.rag.ensemble_retriever import EnsembleRetriever
from utils.add_documents_to_rag import load_true_news_docs  # if you created it

load_dotenv()


def _format_docs(docs: List[Document]) -> str:
    """Format documents for LLM context."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"Document {i}:\n{doc.page_content}")
    return "\n\n".join(formatted)


class HybridRAGSystem:
    """
    Hybrid RAG system (dense + BM25 + Flashrank compression),
    modeled like GraphRAGSystem.

    - Build once in __init__
    - Use .retrieve(question) to get reranked docs
    - Use .query(question) to get a final answer string
    """

    def __init__(
        self,
        csv_path: str = "src/data/News_dataset/True.csv",
        limit_docs: Optional[int] = 3,
        persist_directory: str = "vectorstore",
        dense_k: int = 5,
        keyword_k: int = 5,
    ) -> None:
        self.csv_path = csv_path
        self.limit_docs = limit_docs
        self.persist_directory = persist_directory
        self.dense_k = dense_k
        self.keyword_k = keyword_k

        # Core models
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Build retrieval stack and QA chain once
        self._build_retriever()
        self._build_chain()

    def _build_retriever(self) -> None:
        # 1) Load documents (from helper or directly)
        documents = load_true_news_docs(self.csv_path, self.limit_docs)

        # 2) Semantic chunking
        splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="gradient",
            breakpoint_threshold_amount=0.8,
        )
        chunks = splitter.split_documents(documents)

        # 3) Dense retriever via Chroma
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        dense_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.dense_k},
        )

        # 4) Keyword retriever (BM25)
        keyword_retriever = BM25Retriever.from_documents(
            documents=chunks,
            k=self.keyword_k,
        )

        # 5) Ensemble retriever (your RRF implementation)
        ensemble = EnsembleRetriever(
            retrievers=[dense_retriever, keyword_retriever],
            weights=[0.5, 0.5],
        )
        ensemble_runnable = RunnableLambda(lambda q: ensemble.invoke(q))

        # 6) Flashrank reranker + contextual compression
        compressor = FlashrankRerank()
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_runnable,
        )

    def _build_chain(self) -> None:
        # Same idea as your current script, just as a method
        prompt = ChatPromptTemplate.from_template(
            """Use the following pieces of context to answer the question at the end.
If you don't know the answer, say that you don't know.
Context: {context}
Question: {question}"""
        )

        compressed_runnable = RunnableLambda(
            lambda q: self.retriever.invoke(q)
        )

        self.chain = (
            {
                "context": compressed_runnable | _format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    # ---------------- Public methods (like GraphRAGSystem) ------------------

    def retrieve(self, question: str) -> List[Document]:
        """
        Use the hybrid retriever (ensemble + Flashrank) to get reranked docs.
        """
        return self.retriever.invoke(question)

    def query(self, question: str) -> str:
        """
        Run full hybrid RAG (retrieve + LLM) and return answer text.
        """
        return self.chain.invoke(question)