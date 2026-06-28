"""
Explanation Generator — produces human-readable reasoning for each ranked candidate.

Architecture position:
  HybridScore  + FeatureVector
        ↓
  ExplanationGenerator.generate()
        ↓
  reasoning: str   (stored in SubmissionRow.reasoning)

Responsibilities:
  - Compose a concise, professional explanation (2–4 sentences) for why a
    candidate was ranked at their position
  - Reference concrete evidence: matched skills, years of experience,
    education tier, certifications, Redrob signals
  - Flag honeypot exclusion reason when applicable
  - Stay within 500 characters (fits CSV cells and UI cards)

Design philosophy:
  - Template-based (no LLM call) — deterministic, fast, auditable
  - Evidence-first: every sentence references a measurable signal
  - Structured into four sections:
      1. Overall fit sentence (score + rank context)
      2. Technical skills evidence
      3. Experience + education evidence
      4. Platform engagement evidence
  - Graceful degradation: if a section has no data, it is omitted
"""

from __future__ import annotations

from src.schemas.scoring import FeatureVector, HybridScore

# ---------------------------------------------------------------------------
# Redrob engagement tier thresholds
# ---------------------------------------------------------------------------

_ENGAGEMENT_HIGH = 0.70
_ENGAGEMENT_MED = 0.45


class ExplanationGenerator:
    """
    Generates deterministic, template-based reasoning strings.

    Stateless — instantiate once and call generate() per candidate.

    Usage:
        gen = ExplanationGenerator()
        reasoning = gen.generate(hybrid_score, feature_vector, rank=1)
    """

    def generate(
        self,
        score: HybridScore,
        fv: FeatureVector,
        rank: int,
    ) -> str:
        """
        Produce a reasoning string for one ranked candidate.

        Args:
            score:  HybridScore with all component scores and final_score.
            fv:     FeatureVector with matched skills, education, signals.
            rank:   Final submission rank (1-indexed).

        Returns:
            A non-empty string of ≤ 500 characters suitable for the CSV
            ``reasoning`` column.
        """
        parts: list[str] = []

        # Section 1: Overall fit
        parts.append(self._overall(score, rank))

        # Section 2: Technical skills
        skills_sentence = self._skills(fv)
        if skills_sentence:
            parts.append(skills_sentence)

        # Section 3: Experience + education
        parts.append(self._exp_edu(fv))

        # Section 4: Platform engagement
        eng_sentence = self._engagement(fv)
        if eng_sentence:
            parts.append(eng_sentence)

        reasoning = " ".join(parts)
        # Hard cap at 500 chars — trim at word boundary
        if len(reasoning) > 500:
            reasoning = reasoning[:497].rsplit(" ", 1)[0] + "..."
        return reasoning

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    @staticmethod
    def _overall(score: HybridScore, rank: int) -> str:
        """Rank and composite score context."""
        pct = round(score.final_score, 1)
        sem_pct = round(score.semantic_similarity * 100, 0)
        return (
            f"Ranked #{rank} with a composite talent score of {pct}/100 "
            f"({sem_pct:.0f}% semantic alignment with the JD)."
        )

    @staticmethod
    def _skills(fv: FeatureVector) -> str:
        """Matched technical skills."""
        must = fv.matched_must_have_skills
        nice = fv.matched_nice_to_have_skills
        certs = fv.cert_names

        parts: list[str] = []
        if must:
            skill_str = ", ".join(must[:5])
            parts.append(f"Matched {len(must)} required skill(s): {skill_str}.")
        if nice:
            parts.append(f"Also demonstrates {len(nice)} preferred skill(s): {', '.join(nice[:3])}.")
        if certs:
            parts.append(f"Holds {len(certs)} relevant certification(s) including {certs[0]}.")

        return " ".join(parts)

    @staticmethod
    def _exp_edu(fv: FeatureVector) -> str:
        """Experience and education evidence."""
        yoe = fv.years_of_experience
        exp_score = round(fv.experience_score * 100)
        degree = fv.highest_education_degree or "unspecified degree"
        edu_score = round(fv.education_score * 100)
        return (
            f"Brings {yoe:.0f} years of experience (experience fit: {exp_score}%) "
            f"with a {degree} (education score: {edu_score}%)."
        )

    @staticmethod
    def _engagement(fv: FeatureVector) -> str:
        """Redrob platform engagement signals."""
        sig = fv.redrob_signal_score

        if sig >= _ENGAGEMENT_HIGH:
            tier = "strong"
        elif sig >= _ENGAGEMENT_MED:
            tier = "moderate"
        else:
            tier = "limited"

        open_flag = "open to opportunities" if fv.signal_open_to_work >= 1.0 else "not currently open"
        response = round(fv.signal_response_rate * 100)

        return (
            f"Platform engagement is {tier} (signal score: {round(sig * 100)}%); "
            f"candidate is {open_flag} with a {response}% recruiter response rate."
        )

    def generate_honeypot(self, candidate_id: str, rank: int) -> str:
        """
        Produce a reasoning string for a honeypot candidate that was excluded.

        This should never appear in the final submission (honeypots are excluded),
        but is provided for audit trail purposes.
        """
        return (
            f"Candidate {candidate_id} was excluded from the final ranking (rank #{rank} "
            f"position unused): detected as a synthetic test profile due to statistically "
            f"impossible platform signal combinations."
        )
