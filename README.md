# StrataSearch

**Documentation search engine specialized in navigating legacy vs modern codebases**

StrataSearch uses an Advanced RAG pipeline to understand code evolution, automatically distinguishing between deprecated patterns and modern best practices in its responses

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Django](https://img.shields.io/badge/Django-5.0-green)
![LangChain](https://img.shields.io/badge/LangChain-0.1-orange)

## Architecture

StrataSearch implements a multi-stage RAG pipeline designed for high-precision technical retrieval

### 1. Ingestion Layer

- **Metadata Scraper**: Custom crawler saves data as structured `JSON` + `TXT`, preserving document titles and URLs for accurate citation
- **Recursive Chunking**: Splits text while respecting code block boundaries and paragraph structure
- **Vectorization**: Uses `FAISS` with `Qwen3-Embedding` for dense semantic indexing

### 2. Retrieval Layer

- **Query Translation (Multi-Query):** Generates 3 semantic variations of the user's query to bridge the vocabulary gap between legacy and modern terms (e.g., "save" vs "commit")
- **Re-Ranking (FlashRank):** A Cross-Encoder re-scores the top retrieved documents to filter out irrelevant matches before they reach the LLM

### 3. Generation Layer

- **Context-Aware Prompting**: Dynamic prompts instruct `Qwen-2.5` to explicitly highlight migration paths when version conflicts are detected
- **Strict Citations**: Responses must cite sources using `[Source: filename]` format

---

## Key Features

- **Legacy vs Modern Code Analysis**: Automatically identifies deprecated code patterns and suggests modern alternatives

- **Advanced RAG Pipeline**: Implements a multi-stage retrieval process including multi-query generation and cross-encoder re-ranking for high-precision results

- **Command-Line Driven Workflow**: Includes Django management commands for data scraping, knowledge base ingestion, and RAG pipeline evaluation

- **Private & Local-First**: Runs entirely offline using local models with Ollama, ensuring data privacy and control.

## Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) running locally

### 1. Setup

```bash
git clone https://github.com/yourusername/stratasearch.git
cd stratasearch
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
Create a local environment file by copying the example template.
```bash
copy .env.example .env  # For Windows Command Prompt
# cp .env.example .env  # For macOS/Linux
```
The default values in `.env.example` are configured for local development and should work out of the box. You can edit `.env` to change the models, chunk sizes, or other settings

### 3. Build Knowledge Base

A. Scrape Documentation

```bash
# Scrape target docs (e.g., SQLAlchemy v2.0, 2 levels deep, max 20 pages)
python manage.py scrape https://docs.sqlalchemy.org/en/20/ --depth 2 --max 20

```

B. Ingest & Index

```bash
python manage.py ingest

```

### 4. Run Server

```bash
python manage.py runserver
```

## 🧪 Evaluation

The project includes a built-in evaluation command using the RAGAs framework to measure the performance of the RAG pipeline

```bash
python manage.py evaluate
```

## 🛠️ Tech Stack

| Component | Technology | Description |

|---|---|---|

| Backend | Django 5.0 | Async Views & Management Commands |

| Pipeline | LangChain | RAG Orchestration |

| Inference | Ollama | Local LLM (Qwen 2.5) |

| Search | FAISS + FlashRank | Vector Search + Cross-Encoder Re-ranking |

| Frontend | HTMX + Tailwind | Reactive UI without heavy frameworks |
