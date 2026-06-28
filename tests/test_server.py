"""
Unit tests for src.api.server

All tests mock the AppState so no model loading, no file I/O, no network.
Uses FastAPI TestClient via httpx.

Covers:
  - GET /health: returns 200 with correct schema
  - POST /rank: valid request → 200 with top_100 list
  - POST /rank: JD too short → 422 validation error
  - GET /top100: no cache → 404
  - GET /top100: after rank → 200 list
  - RankRequest field validation (retrieval_top_k bounds)
  - CandidateScore fields present and in range
"""

from __future__ import annotations

import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.server import app, _state
from src.reranker.cross_encoder import RerankResult
from src.retrieval.retriever import RetrievalResult
from src.schemas.scoring import FeatureVector, HybridScore
from src.scoring.hybrid_scorer import ScoringResult

# ---------------------------------------------------------------------------
# Fixtures — synthetic pipeline objects
# ---------------------------------------------------------------------------

TODAY = datetime.date.today()


def _make_fv(cid: str) -> FeatureVector:
    return FeatureVector(
        candidate_id=cid,
        experience_score=0.8,
        education_score=0.75,
        certification_score=0.3,
        redrob_signal_score=0.65,
        signal_open_to_work=1.0,
        signal_response_rate=0.85,
        signal_interview_completion=0.9,
        signal_profile_completeness=0.8,
        signal_recency=0.95,
        signal_github=0.55,
        signal_assessment_avg=0.65,
        signal_saved_by_recruiters=0.3,
        is_honeypot=False,
        years_of_experience=6.0,
        highest_education_degree="B.Tech",
        matched_must_have_skills=["python", "faiss"],
        matched_nice_to_have_skills=["kubernetes"],
        cert_names=["AWS ML Specialty"],
    )


def _make_scoring_result(n: int = 100) -> ScoringResult:
    ids = [f"CAND_{i:07d}" for i in range(1, n + 1)]
    ranked = [
        HybridScore.compute(
            candidate_id=cid,
            semantic_similarity=0.85,
            cross_encoder_score=0.70,
            experience_score=0.80,
            redrob_signal_score=0.65,
            education_score=0.75,
            certification_score=0.30,
        )
        for cid in ids
    ]
    return ScoringResult(
        ranked=ranked,
        all_scored=ranked,
        honeypots=[],
        num_input=n,
    )


def _patch_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch AppState with mocks so no model loading occurs."""
    n = 100
    ids = [f"CAND_{i:07d}" for i in range(1, n + 1)]

    # Mock retriever
    mock_retriever = MagicMock()
    mock_retriever.num_candidates = 100_000
    mock_retriever.retrieve.return_value = RetrievalResult(
        candidate_ids=ids,
        semantic_scores=np.array([0.85] * n, dtype=np.float32),
        query_text="test jd",
        top_k=n,
        retrieval_time_ms=5.0,
    )

    # Mock reranker
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = RerankResult(
        candidate_ids=ids,
        semantic_scores=np.array([0.85] * n, dtype=np.float32),
        ce_scores=np.array([0.70] * n, dtype=np.float32),
        ce_raw_scores=np.array([1.5] * n, dtype=np.float32),
        query_text="test jd",
        rerank_time_ms=200.0,
    )

    # Mock feature engineer
    mock_fe = MagicMock()
    mock_fe.extract.side_effect = lambda profile, jd: _make_fv(profile.candidate_id)

    # Mock scorer
    mock_scorer = MagicMock()
    mock_scorer.score_all.return_value = _make_scoring_result(n)

    # Mock explainer
    mock_explainer = MagicMock()
    mock_explainer.generate.return_value = "Excellent candidate with strong AI/ML skills."

    # Mock JD parser
    from src.schemas.jd import (
        ExperienceLevel, ExperienceRequirement, LocationRequirement,
        StructuredJD,
    )
    mock_jd = StructuredJD(
        title="Senior AI Engineer",
        company="Test Co",
        raw_text="Senior AI engineer with FAISS experience required.",
        must_have_skills=[],
        nice_to_have_skills=[],
        disqualifying_patterns=[],
        experience=ExperienceRequirement(
            min_years=4.0, max_years=10.0,
            preferred_level=ExperienceLevel.SENIOR,
        ),
        location=LocationRequirement(),
        key_technologies=["python", "faiss"],
        embedding_text="Senior AI engineer with FAISS experience.",
    )
    mock_jd_parser = MagicMock()
    mock_jd_parser.from_raw_text.return_value = mock_jd

    # Mock candidate objects (needed for feature extraction lookup)
    mock_profiles = {}
    for cid in ids:
        p = MagicMock()
        p.candidate_id = cid
        mock_profiles[cid] = p

    _state.retriever = mock_retriever
    _state.reranker = mock_reranker
    _state.feature_engineer = mock_fe
    _state.scorer = mock_scorer
    _state.explainer = mock_explainer
    _state.jd_parser = mock_jd_parser
    _state.candidate_texts = {cid: f"Candidate text {cid}" for cid in ids}
    _state.candidate_map = mock_profiles
    _state.last_result = None
    _state.startup_time_s = 1.2


# ---------------------------------------------------------------------------
# Client fixture (no lifespan — state patched directly)
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    _patch_state(monkeypatch)
    # Use transport="asgi" without triggering lifespan
    return TestClient(app, raise_server_exceptions=True)


VALID_JD = (
    "We are looking for a Senior AI / ML Engineer with 5+ years of experience "
    "building production-grade NLP and retrieval systems. Must have: Python, FAISS, "
    "sentence-transformers, cross-encoders, PyTorch. Nice to have: Kubernetes, MLOps."
)

# ---------------------------------------------------------------------------
# Tests — /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_200(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_num_candidates_in_response(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["num_candidates"] == 100_000

    def test_startup_time_present(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "startup_time_s" in data


# ---------------------------------------------------------------------------
# Tests — POST /rank
# ---------------------------------------------------------------------------


class TestRank:
    def test_returns_200(self, client: TestClient) -> None:
        r = client.post("/rank", json={"jd_text": VALID_JD})
        assert r.status_code == 200

    def test_top_100_length(self, client: TestClient) -> None:
        data = client.post("/rank", json={"jd_text": VALID_JD}).json()
        assert len(data["top_100"]) == 100

    def test_has_pipeline_time(self, client: TestClient) -> None:
        data = client.post("/rank", json={"jd_text": VALID_JD}).json()
        assert data["pipeline_time_ms"] >= 0

    def test_candidate_id_format(self, client: TestClient) -> None:
        data = client.post("/rank", json={"jd_text": VALID_JD}).json()
        for entry in data["top_100"]:
            assert entry["candidate_id"].startswith("CAND_")

    def test_ranks_sequential(self, client: TestClient) -> None:
        data = client.post("/rank", json={"jd_text": VALID_JD}).json()
        ranks = [e["rank"] for e in data["top_100"]]
        assert ranks == list(range(1, 101))

    def test_scores_in_range(self, client: TestClient) -> None:
        data = client.post("/rank", json={"jd_text": VALID_JD}).json()
        for e in data["top_100"]:
            assert 0.0 <= e["final_score"] <= 100.0

    def test_reasoning_non_empty(self, client: TestClient) -> None:
        data = client.post("/rank", json={"jd_text": VALID_JD}).json()
        for e in data["top_100"]:
            assert len(e["reasoning"].strip()) > 0

    def test_short_jd_returns_422(self, client: TestClient) -> None:
        r = client.post("/rank", json={"jd_text": "Too short"})
        assert r.status_code == 422

    def test_custom_retrieval_k_accepted(self, client: TestClient) -> None:
        r = client.post(
            "/rank", json={"jd_text": VALID_JD, "retrieval_top_k": 300, "rerank_top_k": 100}
        )
        assert r.status_code == 200

    def test_retrieval_k_too_small_returns_422(self, client: TestClient) -> None:
        r = client.post("/rank", json={"jd_text": VALID_JD, "retrieval_top_k": 5})
        assert r.status_code == 422

    def test_response_has_num_candidates_indexed(self, client: TestClient) -> None:
        data = client.post("/rank", json={"jd_text": VALID_JD}).json()
        assert data["num_candidates_indexed"] == 100_000


# ---------------------------------------------------------------------------
# Tests — GET /top100
# ---------------------------------------------------------------------------


class TestTop100:
    def test_no_cache_returns_404(self, client: TestClient) -> None:
        r = client.get("/top100")
        assert r.status_code == 404

    def test_after_rank_returns_200(self, client: TestClient) -> None:
        client.post("/rank", json={"jd_text": VALID_JD})
        r = client.get("/top100")
        assert r.status_code == 200

    def test_after_rank_returns_list(self, client: TestClient) -> None:
        client.post("/rank", json={"jd_text": VALID_JD})
        data = client.get("/top100").json()
        assert isinstance(data, list)
        assert len(data) == 100
