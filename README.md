# Agentic-News-Bot

A personalized press agent powered by AI, featuring news generation, fake news detection, and press conference simulation capabilities.

## 🎯 Key Features

- **News Generation**: Automated news creation
- **Fake News Detection**: ML-powered detection to identify unreliable news articles
- **Press Conference Simulator**: Interactive press conference simulation system

> **Note**: At the moment This repository currently dosen't contain the Press Conference Simulator. The feature is yet to be implemented .

## 📁 Project Structure

```bash
Agentic-News-Bot/
├── app.py                          # Flask app: classic fake-news detection API
├── streamlit_app.py                # Streamlit chat UI (multi-agent supervisor)
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (not tracked)
├── .gitignore                      # Git ignore rules
│
├── architecture/                   # Project setup and documentation
│   └── project-structure-script.sh # Script to generate base project structure
│
├── notebooks/                      # Jupyter notebooks for exploration
│   └── fake-news-detection.ipynb  # Fake news detection analysis
│
├── src/                            # Source code
│   ├── agents/                     # AI agents + LangGraph supervisor
│   │   ├── agent.py                # Supervisor graph & routing rules
│   │   ├── content_creator_agent.py # Article generation agent
│   │   ├── analyst_agent.py        # Summarization / sentiment agent
│   │   ├── detector_agent.py       # Fake-news verification agent (ML + LLM)
│   │   ├── rag_agents.py           # Hybrid RAG + Graph RAG agents/tools
│   │   └── news_prediction_agent.py# Classic ML prediction agent for app.py
│   │
│   ├── data/                       # Datasets
│   │   └── News_dataset/
│   │       ├── Fake.csv            # Fake news samples
│   │       └── True.csv            # True news samples
│   │
│   ├── embeddings/                 # Text embedding models
│   │   └── embed_model.py          # SentenceTransformer wrapper (all-MiniLM)
│   │
│   ├── rag/                        # RAG components (hybrid + graph)
│   │   ├── ensemble_retriever.py   # Dense + BM25 RRF-style ensemble
│   │   ├── hybrid_rag_system.py    # HybridRAGSystem over CSV news corpus
│   │   └── graph_rag_system.py     # GraphRAGSystem over Neo4j (Cypher + QA)
│   │
│   └── models/                     # Trained ML models
│       ├── best_model.pkl          # Best performing model
│       ├── logisticRegressor.pkl   # Logistic regression model
│       ├── minmax_scaler.pkl       # Feature scaler
│       ├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
│       └── embedding_model/        # Pre-trained sentence transformer
│
├── templates/                      # HTML templates
│   └── index.html                  # Web interface for app.py
│
├── tests/                          # Test and evaluation scripts
│   ├── news_prediction.py          # Classic ML prediction smoke test
│   ├── supervisor_test.py          # LangGraph supervisor end-to-end trace
│   ├── rag_test_eval.py            # Graph RAG + ragas evaluation
│   └── hybrid_rag_test_eval.py     # Hybrid RAG + ragas evaluation
│
└── utils/                          # Utility functions and tools
    ├── data_preprocessing.py       # Data cleaning and preprocessing
    ├── data_validation.py          # Pydantic schemas (NewsItem, Verification)
    ├── simulation_helpers.py       # Synthetic news generator via LLM
    ├── tools.py                    # LangChain tools (content/analysis/verify)
    └── train_and_save_model.py     # Model training pipeline
```

### Quick Setup

To recreate the project structure from scratch, run:

```bash
bash architecture/project-structure-script.sh
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/IyedGuezmir/Agentic-News-Bot.git
cd Agentic-News-Bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Running the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 🧠 Fake News Detection

The fake news detection system uses a hybrid approach that combines classic ML, LLM verification, and optional RAG-based context.

### Classic ML + Web Verification (`app.py`)
1. **Text Embedding**: News articles are converted to semantic embeddings using a SentenceTransformer (`all-MiniLM-L6-v2`) via `src/embeddings/embed_model.py`.
2. **ML Prediction**: A pre-trained logistic regression classifier predicts if the news is fake or true with a confidence score.
3. **Web Verification**: An LLM (`gpt-4o-mini`) verifies the news against online sources and returns a structured verdict.
4. **Final Decision**: If web verification finds credible sources, the article is marked as True News; otherwise, the ML model prediction is used.

### Multi-Agent Supervisor + RAG (`streamlit_app.py`)

Beyond the standalone Flask module, the project exposes a full **agentic workflow** via `streamlit_app.py`:
- A LangGraph **supervisor** (`src/agents/agent.py`) routes each user message to exactly one specialized agent:
  - `content_creator` – generates a single news article given a subject/date.
  - `analyst` – either **summarizes** the most recent article or **analyzes sentiment/tone**.
  - `detector` – performs fake-news detection by combining the ML model and an LLM-based web credibility check.
  - `hybrid_rag` – answers corpus-style questions over the CSV news dataset using `HybridRAGSystem` (dense + BM25 + reranker).
  - `graph_rag` – answers relationship/entity questions over a Neo4j news graph using `GraphRAGSystem`.
- This lets you run a full workflow in one conversation:
  1. **Generate** an article.
  2. **Summarize** or **analyze sentiment**.
  3. **Verify** authenticity.
  4. Ask broader **RAG questions** about patterns, entities, or history in the corpus/graph.

### Key Components
- **Sentence Transformers** (`all-MiniLM-L6-v2`): For semantic text embeddings.
- **Pre-trained ML Classifier**: For initial fake-news prediction.
- **LangChain + OpenAI GPT-4o-mini**: For content creation, analysis, verification, and RAG reasoning.
- **Hybrid Decision Logic**: Combines ML predictions, web verification, and RAG context across the whole agentic workflow.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
