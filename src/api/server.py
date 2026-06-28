"""
FastAPI application for the AI Talent Intelligence Platform.

Provides two sets of endpoints:

  /health          — liveness probe
  /rank            — rank candidates for a given JD text (full pipeline)
  /candidates/{id} — fetch a single candidate's scores (for UI drill-down)
  /top100          — return the latest cached top-100 ranking

Architecture:
  - Application state is held in AppState (loaded once at startup via lifespan)
  - All heavy I/O (model loading, FAISS index) happens in the lifespan context
  - Endpoints are thin: validate input → delegate to pipeline → return typed response
  - No global mutable state outside AppState

Pipeline flow per request to /rank:
  JD text
    → JDParser.from_raw_text()
    → Retriever.retrieve()        (FAISS top-500)
    → CrossEncoderReranker.rerank()
    → FeatureEngineer.extract() × N
    → HybridScorer.score_all()
    → ExplanationGenerator.generate() × 100
    → RankResponse

Environment variables (all optional, have defaults):
  ARTIFACTS_DIR   Path to pre-computed embeddings + FAISS index
                  Default: ./artifacts
  RETRIEVAL_TOP_K Number of candidates retrieved per query
                  Default: 500
  RERANK_TOP_K    Number of candidates after cross-encoder reranking
                  Default: 200
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from src.embeddings.engine import EmbeddingEngine
from src.explanation.generator import ExplanationGenerator
from src.features.feature_engineer import FeatureEngineer
from src.parsers.candidate_parser import CandidateParser, CandidateTextBuilder
from src.parsers.jd_parser import JDParser
from src.reranker.cross_encoder import CrossEncoderReranker
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore
from src.scoring.hybrid_scorer import HybridScorer, ScoringResult
from src.schemas.scoring import HybridScore

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

_ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
_RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "500"))
_RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "200"))
_CANDIDATES_JSONL = Path(os.getenv("CANDIDATES_JSONL", "data/candidates.jsonl"))
_JD_DOCX = Path(os.getenv("JD_DOCX", "data/job_description.docx"))

# ---------------------------------------------------------------------------
# Application state (loaded once at startup)
# ---------------------------------------------------------------------------


class AppState:
    """Container for all heavy pipeline objects loaded at startup."""

    retriever: Retriever
    reranker: CrossEncoderReranker
    feature_engineer: FeatureEngineer
    scorer: HybridScorer
    explainer: ExplanationGenerator
    jd_parser: JDParser
    candidate_texts: dict[str, str]        # candidate_id → embedding text
    candidate_map: dict[str, object]       # candidate_id → CandidateProfile
    last_result: ScoringResult | None = None
    startup_time_s: float = 0.0


_state = AppState()

# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load all heavy pipeline artifacts before serving requests."""
    t0 = time.perf_counter()
    logger.info("=== AI Talent Intelligence Platform — startup ===")

    # 1. Retriever (loads bge-small-en-v1.5 + FAISS index)
    logger.info(f"Loading Retriever from {_ARTIFACTS_DIR} ...")
    _state.retriever = Retriever.from_artifacts(
        _ARTIFACTS_DIR,
        default_top_k=_RETRIEVAL_TOP_K,
    )

    # 2. Cross-encoder
    logger.info("Loading CrossEncoderReranker ...")
    _state.reranker = CrossEncoderReranker()
    _state.reranker.load_model()

    # 3. Stateless pipeline objects
    _state.feature_engineer = FeatureEngineer()
    _state.scorer = HybridScorer()
    _state.explainer = ExplanationGenerator()
    _state.jd_parser = JDParser()

    # 4. Build candidate text map (needed for cross-encoder pairs)
    logger.info(f"Streaming candidate texts from {_CANDIDATES_JSONL} ...")
    builder = CandidateTextBuilder()
    parser = CandidateParser(_CANDIDATES_JSONL)
    candidate_texts: dict[str, str] = {}
    candidate_map: dict[str, object] = {}
    for profile in parser.stream():
        candidate_texts[profile.candidate_id] = builder.build(profile)
        candidate_map[profile.candidate_id] = profile
    _state.candidate_texts = candidate_texts
    _state.candidate_map = candidate_map
    logger.info(f"Loaded {len(candidate_texts):,} candidate text representations.")

    _state.startup_time_s = time.perf_counter() - t0
    logger.info(f"Startup complete in {_state.startup_time_s:.1f}s")

    yield  # Application runs here

    logger.info("=== Shutting down ===")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Talent Intelligence Platform",
    description=(
        "Production-grade candidate ranking API. "
        "Ranks 100K+ candidates using FAISS retrieval + cross-encoder reranking "
        "+ hybrid scoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class RankRequest(BaseModel):
    """Input for the /rank endpoint."""

    jd_text: str = Field(
        ...,
        min_length=50,
        description="Raw job description text. Minimum 50 characters.",
        examples=["Senior AI Engineer with FAISS, NLP, and Python experience required."],
    )
    retrieval_top_k: int = Field(
        default=_RETRIEVAL_TOP_K,
        ge=10,
        le=2000,
        description="Number of candidates retrieved from FAISS before reranking.",
    )
    rerank_top_k: int = Field(
        default=_RERANK_TOP_K,
        ge=10,
        le=1000,
        description="Number of candidates kept after cross-encoder reranking.",
    )


class CandidateScore(BaseModel):
    """Score breakdown for a single ranked candidate."""

    rank: int
    candidate_id: str
    final_score: float
    semantic_similarity: float
    cross_encoder_score: float
    experience_score: float
    education_score: float
    certification_score: float
    redrob_signal_score: float
    is_honeypot: bool
    reasoning: str


class RankResponse(BaseModel):
    """Full response from the /rank endpoint."""

    top_100: list[CandidateScore]
    num_candidates_indexed: int
    num_honeypots_excluded: int
    retrieval_top_k: int
    rerank_top_k: int
    pipeline_time_ms: float


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str
    num_candidates: int
    startup_time_s: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness probe — returns 200 when the pipeline is ready."""
    return HealthResponse(
        status="ok",
        num_candidates=_state.retriever.num_candidates,
        startup_time_s=round(_state.startup_time_s, 2),
    )


@app.post("/rank", response_model=RankResponse, tags=["ranking"])
async def rank(req: RankRequest) -> RankResponse:
    """
    Run the full ranking pipeline for a job description.

    Pipeline steps:
      1. Parse JD text → StructuredJD
      2. FAISS retrieval → top-K candidates
      3. Cross-encoder reranking → top-K' candidates
      4. Feature extraction per candidate
      5. Hybrid scoring → ScoringResult
      6. Explanation generation → reasoning strings
    """
    t0 = time.perf_counter()

    # Step 1: Parse JD
    try:
        jd = _state.jd_parser.from_raw_text(req.jd_text)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"JD parsing failed: {exc}") from exc

    # Step 2: Retrieve
    retrieval = _state.retriever.retrieve(jd, top_k=req.retrieval_top_k)

    # Step 3: Rerank
    reranked = _state.reranker.rerank(
        retrieval,
        _state.candidate_texts,
        top_k=req.rerank_top_k,
    )

    # Step 4: Feature extraction
    feature_vectors = []
    for cid in reranked.candidate_ids:
        profile = _state.candidate_map.get(cid)
        if profile is None:
            continue
        fv = _state.feature_engineer.extract(profile, jd)  # type: ignore[arg-type]
        feature_vectors.append(fv)

    if not feature_vectors:
        raise HTTPException(
            status_code=500,
            detail="No feature vectors could be computed — check candidate data.",
        )

    # Step 5: Hybrid scoring
    scoring_result = _state.scorer.score_all(
        reranked, feature_vectors, top_n=100
    )
    _state.last_result = scoring_result

    # Step 6: Explanations + assemble response
    scores: list[CandidateScore] = []
    for rank_idx, hs in enumerate(scoring_result.ranked, start=1):
        fv_map = {fv.candidate_id: fv for fv in feature_vectors}
        fv = fv_map.get(hs.candidate_id)
        reasoning = (
            _state.explainer.generate(hs, fv, rank=rank_idx)  # type: ignore[arg-type]
            if fv
            else f"Ranked #{rank_idx} with score {hs.final_score:.2f}."
        )
        scores.append(
            CandidateScore(
                rank=rank_idx,
                candidate_id=hs.candidate_id,
                final_score=round(hs.final_score, 4),
                semantic_similarity=round(hs.semantic_similarity, 4),
                cross_encoder_score=round(hs.cross_encoder_score, 4),
                experience_score=round(hs.experience_score, 4),
                education_score=round(hs.education_score, 4),
                certification_score=round(hs.certification_score, 4),
                redrob_signal_score=round(hs.redrob_signal_score, 4),
                is_honeypot=hs.is_honeypot,
                reasoning=reasoning,
            )
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return RankResponse(
        top_100=scores,
        num_candidates_indexed=_state.retriever.num_candidates,
        num_honeypots_excluded=scoring_result.num_honeypots,
        retrieval_top_k=req.retrieval_top_k,
        rerank_top_k=req.rerank_top_k,
        pipeline_time_ms=round(elapsed_ms, 1),
    )


@app.get("/top100", response_model=list[CandidateScore], tags=["ranking"])
async def top100() -> list[CandidateScore]:
    """Return the most recently cached top-100 ranking (populated after a /rank call)."""
    if _state.last_result is None:
        raise HTTPException(
            status_code=404,
            detail="No ranking cached yet. Call POST /rank first.",
        )
    # Re-build response from cached result (no reasoning since no fv_map cached)
    scores = []
    for rank_idx, hs in enumerate(_state.last_result.ranked, start=1):
        scores.append(
            CandidateScore(
                rank=rank_idx,
                candidate_id=hs.candidate_id,
                final_score=round(hs.final_score, 4),
                semantic_similarity=round(hs.semantic_similarity, 4),
                cross_encoder_score=round(hs.cross_encoder_score, 4),
                experience_score=round(hs.experience_score, 4),
                education_score=round(hs.education_score, 4),
                certification_score=round(hs.certification_score, 4),
                redrob_signal_score=round(hs.redrob_signal_score, 4),
                is_honeypot=hs.is_honeypot,
                reasoning=f"Cached score: {hs.final_score:.2f}/100",
            )
        )
    return scores
