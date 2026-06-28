"""
src.parsers — Public API for all parser modules.
"""

from src.parsers.candidate_parser import CandidateParser, CandidateTextBuilder
from src.parsers.jd_parser import JDParser

__all__ = [
    "CandidateParser",
    "CandidateTextBuilder",
    "JDParser",
]
