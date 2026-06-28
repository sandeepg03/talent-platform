"""
Feature Engineer — extracts and normalises per-candidate feature scores.

Architecture position:
  RerankResult (top-K reranked candidates)
        ↓
  FeatureEngineer.extract(candidate, jd)
        ↓
  FeatureVector (stored in src.schemas.scoring)
        ↓
  HybridScorer

Responsibilities:
  - Compute experience_score:    years vs JD requirement band → [0, 1]
  - Compute education_score:     tier + degree level → [0, 1]
  - Compute certification_score: relevant certifications vs JD requirements → [0, 1]
  - Compute redrob_signal_score: weighted composite of 8 behavioural signals → [0, 1]
  - Detect honeypots:            candidates with physically impossible signals
  - Build matched_must_have / matched_nice_to_have skill lists (for explanations)

All sub-scores are normalised to [0, 1] so HybridScorer can apply weights directly.

Honeypot detection heuristics (Stage 3 filter — ~80 profiles):
  ANY ONE of the following disqualifies a candidate as a honeypot:
    1. response_rate == 1.0  AND  avg_response_time_hours < 0.1
       (instantly responds to every recruiter — statistically impossible)
    2. github_activity_score == 100.0 AND profile_completeness == 100.0
       AND interview_completion_rate == 1.0
       (perfect on every signal simultaneously)
    3. offer_acceptance_rate == -1.0  (sentinel for injected dummy data)
    4. notice_period_days == 0 AND open_to_work_flag is False
       (available immediately but not open to work — contradiction)

Note on github_activity_score: the field allows -1.0 as a sentinel for
"no GitHub account". We treat -1.0 as 0 signal (not negative).
"""

from __future__ import annotations

from typing import Sequence

from src.schemas.candidate import (
    CandidateProfile,
    EducationTier,
    SkillProficiency,
)
from src.schemas.jd import ExperienceLevel, StructuredJD
from src.schemas.scoring import FeatureVector

# ---------------------------------------------------------------------------
# Education tier → numeric weight
# ---------------------------------------------------------------------------

_TIER_WEIGHT: dict[EducationTier, float] = {
    EducationTier.TIER_1: 1.0,
    EducationTier.TIER_2: 0.75,
    EducationTier.TIER_3: 0.50,
    EducationTier.TIER_4: 0.25,
    EducationTier.UNKNOWN: 0.15,
}

# Degree level → additive bonus (summed with tier weight, then clamped)
_DEGREE_BONUS: dict[str, float] = {
    "phd": 0.20,
    "ph.d": 0.20,
    "doctorate": 0.20,
    "master": 0.10,
    "m.tech": 0.10,
    "mtech": 0.10,
    "m.sc": 0.10,
    "msc": 0.10,
    "m.e": 0.10,
    "me ": 0.10,
    "mba": 0.05,
    "bachelor": 0.0,
    "b.tech": 0.0,
    "btech": 0.0,
    "b.e": 0.0,
    "be ": 0.0,
    "b.sc": 0.0,
    "diploma": -0.10,
}

# Certification keywords → relevance signal to an AI/ML JD
_RELEVANT_CERT_KEYWORDS: frozenset[str] = frozenset({
    "machine learning",
    "deep learning",
    "nlp",
    "natural language",
    "tensorflow",
    "pytorch",
    "aws",
    "gcp",
    "azure",
    "cloud",
    "data science",
    "data engineering",
    "mlops",
    "kubernetes",
    "docker",
    "python",
    "sql",
    "spark",
    "airflow",
    "databricks",
    "llm",
    "generative ai",
    "ai",
    "neural",
    "computer vision",
})


# ---------------------------------------------------------------------------
# Feature Engineer
# ---------------------------------------------------------------------------


class FeatureEngineer:
    """
    Extracts, weights, and normalises per-candidate sub-scores.

    Stateless — all methods are pure functions over (CandidateProfile, StructuredJD).
    The class is instantiated once and reused across all candidates.

    Usage:
        engineer = FeatureEngineer()
        fv = engineer.extract(candidate, jd)
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extract(
        self,
        candidate: CandidateProfile,
        jd: StructuredJD,
    ) -> FeatureVector:
        """
        Produce a fully populated FeatureVector for one candidate.

        All sub-scores are in [0, 1] after this method returns.

        Args:
            candidate: Parsed CandidateProfile from candidates.jsonl.
            jd:        Parsed StructuredJD containing scoring criteria.

        Returns:
            FeatureVector ready for HybridScorer.score().
        """
        sig = candidate.redrob_signals
        experience_score = self._experience_score(candidate, jd)
        education_score = self._education_score(candidate)
        certification_score = self._certification_score(candidate, jd)

        (
            redrob_signal_score,
            open_to_work,
            response_rate,
            interview_completion,
            profile_completeness,
            recency,
            github,
            assessment_avg,
            saved_by_recruiters,
        ) = self._redrob_score(candidate)

        is_honeypot = self._is_honeypot(candidate)

        must_have_names = {s.name.lower() for s in jd.must_have_skills}
        nice_to_have_names = {s.name.lower() for s in jd.nice_to_have_skills}
        candidate_skill_names = set(candidate.skill_names)

        matched_must = sorted(
            candidate_skill_names & must_have_names
        )
        matched_nice = sorted(
            candidate_skill_names & nice_to_have_names
        )

        cert_names = [c.name for c in candidate.certifications]

        return FeatureVector(
            candidate_id=candidate.candidate_id,
            experience_score=experience_score,
            education_score=education_score,
            certification_score=certification_score,
            redrob_signal_score=redrob_signal_score,
            signal_open_to_work=open_to_work,
            signal_response_rate=response_rate,
            signal_interview_completion=interview_completion,
            signal_profile_completeness=profile_completeness,
            signal_recency=recency,
            signal_github=github,
            signal_assessment_avg=assessment_avg,
            signal_saved_by_recruiters=saved_by_recruiters,
            is_honeypot=is_honeypot,
            years_of_experience=candidate.profile.years_of_experience,
            highest_education_degree=self._highest_degree(candidate),
            matched_must_have_skills=matched_must,
            matched_nice_to_have_skills=matched_nice,
            cert_names=cert_names,
        )

    # ------------------------------------------------------------------
    # Experience score
    # ------------------------------------------------------------------

    def _experience_score(
        self,
        candidate: CandidateProfile,
        jd: StructuredJD,
    ) -> float:
        """
        Score years of experience against JD requirement band.

        Scoring rules:
          - Below min_years: linear ramp from 0 to 1 as years approach min
            (partial credit — a candidate with 4 years vs min=5 is not zero)
          - Within [min, max]: score = 1.0
          - Above max_years: linear decay toward 0.7 (over-qualified, not disqualified)
            capped at min score 0.7 regardless of how over-qualified
          - If min_years is None: treated as 0; if max_years is None: min + 10
        """
        years = candidate.profile.years_of_experience
        exp_req = jd.experience

        lo = exp_req.min_years if exp_req.min_years is not None else 0.0
        hi = exp_req.max_years if exp_req.max_years is not None else lo + 10.0

        if years < lo:
            # Linear ramp: 0 at 0 years, 1.0 at lo years
            score = years / lo if lo > 0.0 else 1.0
        elif years <= hi:
            score = 1.0
        else:
            # Over-qualified: decay from 1.0 toward 0.70 over next 5 years
            excess = years - hi
            score = max(1.0 - (excess / 5.0) * 0.30, 0.70)

        return float(max(0.0, min(1.0, score)))

    # ------------------------------------------------------------------
    # Education score
    # ------------------------------------------------------------------

    def _education_score(self, candidate: CandidateProfile) -> float:
        """
        Score based on highest education tier and degree level.

        Returns the maximum score across all education entries (best degree wins).
        """
        if not candidate.education:
            return 0.10  # No education data — small baseline

        best = 0.0
        for edu in candidate.education:
            tier_weight = _TIER_WEIGHT.get(edu.tier, 0.15)
            degree_bonus = self._degree_bonus(edu.degree)
            score = min(tier_weight + degree_bonus, 1.0)
            best = max(best, score)

        return float(best)

    @staticmethod
    def _degree_bonus(degree: str) -> float:
        degree_lower = degree.lower()
        for keyword, bonus in _DEGREE_BONUS.items():
            if keyword in degree_lower:
                return bonus
        return 0.0

    @staticmethod
    def _highest_degree(candidate: CandidateProfile) -> str:
        if not candidate.education:
            return "unknown"
        return candidate.education[0].degree

    # ------------------------------------------------------------------
    # Certification score
    # ------------------------------------------------------------------

    def _certification_score(
        self,
        candidate: CandidateProfile,
        jd: StructuredJD,
    ) -> float:
        """
        Score based on number and relevance of certifications.

        Scoring:
          - 0 certifications: 0.0
          - Each relevant cert (keyword match against RELEVANT_CERT_KEYWORDS): +0.30
          - Each any-cert (irrelevant): +0.10
          - Capped at 1.0
        """
        if not candidate.certifications:
            return 0.0

        jd_tech_lower = {t.lower() for t in jd.key_technologies}
        combined_relevant = _RELEVANT_CERT_KEYWORDS | jd_tech_lower

        score = 0.0
        for cert in candidate.certifications:
            cert_lower = cert.name.lower()
            is_relevant = any(kw in cert_lower for kw in combined_relevant)
            score += 0.30 if is_relevant else 0.10

        return float(min(score, 1.0))

    # ------------------------------------------------------------------
    # Redrob signal score
    # ------------------------------------------------------------------

    def _redrob_score(
        self,
        candidate: CandidateProfile,
    ) -> tuple[float, float, float, float, float, float, float, float, float]:
        """
        Compute the composite Redrob signal score and its 8 sub-components.

        Sub-component weights (sum to 1.0):
          open_to_work            0.25  (binary — most impactful single signal)
          response_rate           0.20  (recruiter engagement quality)
          interview_completion    0.15  (follow-through commitment)
          profile_completeness    0.15  (data quality / professionalism)
          recency                 0.10  (time since last active)
          github_activity         0.05  (technical presence)
          assessment_avg          0.05  (skill validation)
          saved_by_recruiters     0.05  (social proof)

        Returns:
            Tuple of (composite, open_to_work, response_rate, interview_completion,
                      profile_completeness, recency, github, assessment_avg,
                      saved_by_recruiters)
            All values in [0, 1].
        """
        sig = candidate.redrob_signals

        # 1. Open to work
        open_to_work = 1.0 if sig.open_to_work_flag else 0.0

        # 2. Response rate (already [0, 1])
        response_rate = float(sig.recruiter_response_rate)

        # 3. Interview completion rate (already [0, 1])
        interview_completion = float(sig.interview_completion_rate)

        # 4. Profile completeness [0, 100] → [0, 1]
        profile_completeness = float(sig.profile_completeness_score) / 100.0

        # 5. Recency: days since last active — cap at 90 days for scoring
        import datetime
        today = datetime.date.today()
        days_inactive = max(0, (today - sig.last_active_date).days)
        recency = float(max(0.0, 1.0 - (days_inactive / 90.0)))

        # 6. GitHub activity: -1.0 sentinel means no account → treat as 0
        raw_github = float(sig.github_activity_score)
        github = max(0.0, raw_github / 100.0)

        # 7. Assessment average: mean of all skill assessment scores [0, 100] → [0, 1]
        if sig.skill_assessment_scores:
            assessment_avg = float(
                sum(sig.skill_assessment_scores.values()) /
                len(sig.skill_assessment_scores)
            ) / 100.0
        else:
            assessment_avg = 0.0

        # 8. Saved by recruiters: cap at 10 for normalisation
        saved_raw = float(sig.saved_by_recruiters_30d)
        saved_by_recruiters = float(min(saved_raw / 10.0, 1.0))

        composite = (
            0.25 * open_to_work
            + 0.20 * response_rate
            + 0.15 * interview_completion
            + 0.15 * profile_completeness
            + 0.10 * recency
            + 0.05 * github
            + 0.05 * assessment_avg
            + 0.05 * saved_by_recruiters
        )
        composite = float(max(0.0, min(1.0, composite)))

        return (
            composite,
            open_to_work,
            response_rate,
            interview_completion,
            profile_completeness,
            recency,
            github,
            assessment_avg,
            saved_by_recruiters,
        )

    # ------------------------------------------------------------------
    # Honeypot detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_honeypot(candidate: CandidateProfile) -> bool:
        """
        Detect candidates with impossible platform signals.

        Returns True if ANY honeypot heuristic fires.
        """
        sig = candidate.redrob_signals

        # Rule 1: Instant universal responder
        if (
            sig.recruiter_response_rate >= 1.0
            and sig.avg_response_time_hours < 0.1
        ):
            return True

        # Rule 2: All-perfect signals simultaneously
        if (
            sig.github_activity_score >= 100.0
            and sig.profile_completeness_score >= 100.0
            and sig.interview_completion_rate >= 1.0
        ):
            return True

        # Rule 3: offer_acceptance_rate sentinel value
        if sig.offer_acceptance_rate == -1.0:
            return True

        # Rule 4: Zero notice and not open to work (contradictory signals)
        if (
            sig.notice_period_days == 0
            and not sig.open_to_work_flag
        ):
            return True

        return False
