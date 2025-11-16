import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag.hybrid_rag_system import HybridRAGSystem
from ragas import evaluate
from ragas.metrics import faithfulness
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i + 1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )

hybrid = HybridRAGSystem()
question = "What kind of news was Trump involved in?"
docs = hybrid.retrieve(question)
pretty_print_docs(docs)

print("\n" + "=" * 100 + "\n")
answer = hybrid.query("What kind of news was Trump involved in? answer in bullet points and add sources/dates if available.")
print(answer)


# ---------- RAGAS faithfulness evaluation ----------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

data = {
   "question": [question],
   "answer": [answer],
   "contexts": [[d.page_content for d in docs]],
}

dataset = Dataset.from_dict(data)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

result = evaluate(
    dataset,
    metrics=[faithfulness],
    llm=llm,
    embeddings=embeddings,
)
print(result)