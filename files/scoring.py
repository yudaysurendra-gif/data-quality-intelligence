"""Aggregates profiling + rule results into overall and per-column quality scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .profiler import DatasetProfile
from .rules import RuleResult


@dataclass
class ColumnScore:
    column: str
    completeness: float   # 1 - missing_pct
    uniqueness: float      # unique_pct, informational (not always "good" to maximize)
    validity: float          # pass rate across rules applied to this column
    score: float               # weighted combination
    rules_applied: int = 0

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "completeness": round(self.completeness, 4),
            "uniqueness": round(self.uniqueness, 4),
            "validity": round(self.validity, 4),
            "score": round(self.score, 4),
            "rules_applied": self.rules_applied,
        }


@dataclass
class QualityScore:
    overall: float
    columns: Dict[str, ColumnScore] = field(default_factory=dict)
    grade: str = ""

    def to_dict(self) -> dict:
        return {
            "overall": round(self.overall, 4),
            "grade": self.grade,
            "columns": {name: c.to_dict() for name, c in self.columns.items()},
        }


def _grade_for(score: float) -> str:
    if score >= 0.95:
        return "A"
    if score >= 0.85:
        return "B"
    if score >= 0.70:
        return "C"
    if score >= 0.50:
        return "D"
    return "F"


class QualityScorer:
    """Combines a DatasetProfile and a list of RuleResults into a QualityScore.

    Per-column score = weighted average of:
        - completeness (1 - missing_pct)
        - validity (pass rate of rules applied to that column; 1.0 if no rules)

    Uniqueness is reported for visibility but excluded from the score by
    default, since high or low uniqueness isn't inherently "good" or "bad"
    depending on the column's semantics (e.g. IDs vs. categories).

    Overall score = simple average of per-column scores.
    """

    def __init__(self, completeness_weight: float = 0.5, validity_weight: float = 0.5):
        total = completeness_weight + validity_weight
        self.completeness_weight = completeness_weight / total
        self.validity_weight = validity_weight / total

    def score(self, profile: DatasetProfile, rule_results: List[RuleResult]) -> QualityScore:
        results_by_column: Dict[str, List[RuleResult]] = {}
        for result in rule_results:
            results_by_column.setdefault(result.column, []).append(result)

        column_scores: Dict[str, ColumnScore] = {}
        for name, col_profile in profile.columns.items():
            completeness = 1.0 - col_profile.missing_pct
            uniqueness = col_profile.unique_pct

            col_rule_results = results_by_column.get(name, [])
            if col_rule_results:
                # validity = average "pass rate" across rules applied to this column
                pass_rates = [1.0 - r.failed_pct for r in col_rule_results]
                validity = sum(pass_rates) / len(pass_rates)
            else:
                validity = 1.0  # no rules defined -> don't penalize

            score = (
                completeness * self.completeness_weight
                + validity * self.validity_weight
            )

            column_scores[name] = ColumnScore(
                column=name,
                completeness=completeness,
                uniqueness=uniqueness,
                validity=validity,
                score=score,
                rules_applied=len(col_rule_results),
            )

        overall = (
            sum(c.score for c in column_scores.values()) / len(column_scores)
            if column_scores else 0.0
        )

        return QualityScore(overall=overall, columns=column_scores, grade=_grade_for(overall))
