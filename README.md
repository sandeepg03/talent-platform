# AI Talent Intelligence Platform

[![CI](https://github.com/sandeepg03/talent-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/sandeepg03/talent-platform/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-grade AI candidate ranking system. Ranks 100,000+ candidates the way an expert recruiter would — using semantic retrieval, cross-encoder reranking, and a transparent weighted scoring formula.

---

## Architecture

```
Job Description
  → JD Parser (StructuredJD)
  → EmbeddingEngine (bge-small-en-v1.5)
  → FAISS IndexFlatIP — Top-500 retrieval
  → CrossEncoder reranking (ms-marco-MiniLM-L-6-v2) — Top-200
  → FeatureEngineer (experience · education · certs · Redrob signals · honeypot detection)
  → HybridScorer (weighted formula)
  → ExplanationGenerator (deterministic reasoning)
  → submission.csv
```

### Scoring Formula

```
Final Score = 0.40 × semantic_similarity
            + 0.30 × cross_encoder_score
            + 0.10 × experience_score
            + 0.10 × redrob_signal_score
            + 0.05 × education_score
            + 0.05 × certification_score

Final Score ∈ [0, 100]
```

---

## Tech Stack

| Component | Library |
|---|---|
| Bi-encoder | `sentence-transformers` (BAAI/bge-small-en-v1.5) |
| ANN search | `faiss-cpu` (IndexFlatIP) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Schema validation | `pydantic` v2 |
| API | `fastapi` + `uvicorn` |
| Dashboard | `streamlit` + `plotly` |
| Logging | `loguru` |
| Tests | `pytest` + `pytest-cov` |
| Container | `Docker` + `docker-compose` |

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/sandeepg03/talent-platform.git
cd talent-platform
pip install -e .
```

### 2. Place data files

```
data/
  candidates.jsonl
  job_description.docx
```

### 3. Precompute embeddings (one-time, ~10 min on CPU)

```bash
python precompute.py \
    --candidates data/candidates.jsonl \
    --artifacts  artifacts/
```

### 4. Run ranking → submission.csv

```bash
python rank.py \
    --jd         data/job_description.docx \
    --candidates data/candidates.jsonl \
    --artifacts  artifacts/ \
    --output     submission.csv
```

### 5. Validate

```bash
python validate_submission.py submission.csv
```

---

## Running with Docker

```bash
# Build
docker build -t talent-platform .

# Precompute (one-time)
docker compose --profile precompute up precompute

# Start API + Streamlit dashboard
docker compose up api ui
```

API: http://localhost:8000  
Dashboard: http://localhost:8501  
API docs: http://localhost:8000/docs

---

## Running Tests

```bash
# All unit tests (fast, no model downloads)
python -m pytest tests/ -m "not integration" -q

# Integration tests (downloads models ~200MB)
python -m pytest tests/ -m "integration" -v

# Coverage report
python -m pytest tests/ -m "not integration" --cov=src --cov-report=term-missing
```

---

## Project Structure

```
talent_platform/
├── src/
│   ├── schemas/          # Pydantic models: CandidateProfile, StructuredJD, HybridScore
│   ├── parsers/          # CandidateParser, JDParser, CandidateTextBuilder
│   ├── embeddings/       # EmbeddingEngine (bge-small-en-v1.5)
│   ├── retrieval/        # VectorStore (FAISS), Retriever
│   ├── reranker/         # CrossEncoderReranker
│   ├── features/         # FeatureEngineer (exp · edu · cert · signals · honeypot)
│   ├── scoring/          # HybridScorer, ScoringResult
│   ├── explanation/      # ExplanationGenerator
│   ├── evaluation/       # RankingEvaluator (NDCG, MRR, P@k)
│   └── api/              # FastAPI server
├── tests/                # 360+ unit + integration tests
├── ui/                   # Streamlit dashboard
├── precompute.py         # Phase 1: build FAISS index
├── rank.py               # Phase 2: run pipeline → submission.csv
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Honeypot Detection

The dataset contains ~80 synthetic test profiles injected to detect keyword-matching systems. The `FeatureEngineer` detects them using 4 heuristics:

| Rule | Condition |
|---|---|
| Instant universal responder | response_time_hours < 0.1 AND response_rate = 1.0 AND offers = 10 |
| Perfect signal constellation | All 6 signals at exactly their maximum values |
| Impossible offer acceptance | offer_acceptance_rate ≥ 0.99 AND notice_period_days = 0 |
| Zero notice + not open | notice_period_days = 0 AND open_to_work = False |

Honeypots are **scored but excluded** from the top-100 submission.

---

## API Reference

### `POST /rank`

```json
{
  "jd_text": "Senior ML Engineer with FAISS, PyTorch...",
  "retrieval_top_k": 500,
  "rerank_top_k": 200
}
```

Response: `{ "top_100": [...], "pipeline_time_ms": 18420, "num_honeypots_excluded": 3 }`

### `GET /health`

```json
{ "status": "ok", "num_candidates": 100000, "startup_time_s": 45.2 }
```

### `GET /top100`

Returns the most recently cached top-100 ranking.

---

## Performance

| Step | Typical Time (CPU) |
|---|---|
| precompute.py (100K candidates) | 8–12 min |
| FAISS retrieval (top-500) | < 5 ms |
| Cross-encoder reranking (500→200) | 8–15 s |
| Feature extraction (200 candidates) | < 1 s |
| Hybrid scoring + explanations | < 1 s |
| **Total rank.py** | **< 5 min** |

---

## License

MIT
