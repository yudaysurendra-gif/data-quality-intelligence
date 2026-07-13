"""Configurable validation rules and the engine that runs them against data."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

import pandas as pd


@dataclass
class RuleResult:
    """Outcome of running a single rule against a column."""

    rule_name: str
    column: str
    passed: bool
    checked: int
    failed: int
    failed_pct: float
    sample_failures: List[Any] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "rule": self.rule_name,
            "column": self.column,
            "passed": self.passed,
            "checked": self.checked,
            "failed": self.failed,
            "failed_pct": round(self.failed_pct, 4),
            "sample_failures": self.sample_failures,
            "message": self.message,
        }


class Rule:
    """Base class for a validation rule applied to one column.

    Subclass and implement `_is_valid(value) -> bool`, or set `name` and
    override `evaluate` entirely for row-level / cross-column rules.
    """

    name: str = "Rule"

    def __init__(self, column: str, *, severity: str = "error", sample_size: int = 5):
        self.column = column
        self.severity = severity  # "error" | "warning" — informational, doesn't change scoring by default
        self.sample_size = sample_size

    def _is_valid(self, value: Any) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def evaluate(self, df: pd.DataFrame) -> RuleResult:
        if self.column not in df.columns:
            return RuleResult(
                rule_name=self.name,
                column=self.column,
                passed=False,
                checked=0,
                failed=0,
                failed_pct=0.0,
                message=f"Column {self.column!r} not found in data.",
            )

        series = df[self.column]
        mask_valid = series.apply(self._is_valid)
        failed_mask = ~mask_valid
        failed_count = int(failed_mask.sum())
        checked = len(series)
        failed_pct = (failed_count / checked) if checked else 0.0

        samples = series[failed_mask].head(self.sample_size).tolist()

        return RuleResult(
            rule_name=self.name,
            column=self.column,
            passed=failed_count == 0,
            checked=checked,
            failed=failed_count,
            failed_pct=failed_pct,
            sample_failures=samples,
        )


class NotNullRule(Rule):
    """Fails for any missing (NaN/None) value."""

    name = "not_null"

    def _is_valid(self, value: Any) -> bool:
        return not pd.isna(value)


class UniqueRule(Rule):
    """Fails for any duplicated value in the column (first occurrence passes)."""

    name = "unique"

    def evaluate(self, df: pd.DataFrame) -> RuleResult:
        if self.column not in df.columns:
            return super().evaluate(df)
        series = df[self.column]
        dup_mask = series.duplicated(keep="first") & series.notna()
        failed_count = int(dup_mask.sum())
        checked = len(series)
        samples = series[dup_mask].head(self.sample_size).tolist()
        return RuleResult(
            rule_name=self.name,
            column=self.column,
            passed=failed_count == 0,
            checked=checked,
            failed=failed_count,
            failed_pct=(failed_count / checked) if checked else 0.0,
            sample_failures=samples,
        )


class RangeRule(Rule):
    """Fails for numeric values outside [min_value, max_value] (inclusive)."""

    name = "range"

    def __init__(self, column: str, min_value: Optional[float] = None,
                 max_value: Optional[float] = None, **kwargs):
        super().__init__(column, **kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def _is_valid(self, value: Any) -> bool:
        if pd.isna(value):
            return True  # NotNullRule's job to catch missing values
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


class RegexRule(Rule):
    """Fails for string values that don't match `pattern`."""

    name = "regex"

    def __init__(self, column: str, pattern: str, **kwargs):
        super().__init__(column, **kwargs)
        self.pattern = re.compile(pattern)

    def _is_valid(self, value: Any) -> bool:
        if pd.isna(value):
            return True
        return bool(self.pattern.match(str(value)))


class AllowedValuesRule(Rule):
    """Fails for values not present in `allowed`."""

    name = "allowed_values"

    def __init__(self, column: str, allowed: Sequence[Any], **kwargs):
        super().__init__(column, **kwargs)
        self.allowed = set(allowed)

    def _is_valid(self, value: Any) -> bool:
        if pd.isna(value):
            return True
        return value in self.allowed


class TypeRule(Rule):
    """Fails for values that aren't an instance of `expected_type`."""

    name = "type"

    def __init__(self, column: str, expected_type: type, **kwargs):
        super().__init__(column, **kwargs)
        self.expected_type = expected_type

    def _is_valid(self, value: Any) -> bool:
        if pd.isna(value):
            return True
        return isinstance(value, self.expected_type)


class CustomRule(Rule):
    """Wraps an arbitrary `value -> bool` predicate as a Rule."""

    def __init__(self, column: str, predicate: Callable[[Any], bool],
                 name: str = "custom", **kwargs):
        super().__init__(column, **kwargs)
        self.name = name
        self._predicate = predicate

    def _is_valid(self, value: Any) -> bool:
        if pd.isna(value):
            return True
        return self._predicate(value)


class RuleEngine:
    """Runs a collection of Rules against a DataFrame and collects results."""

    def __init__(self, rules: List[Rule]):
        self.rules = rules

    def run(self, df: pd.DataFrame) -> List[RuleResult]:
        return [rule.evaluate(df) for rule in self.rules]

    def add(self, rule: Rule) -> "RuleEngine":
        self.rules.append(rule)
        return self
