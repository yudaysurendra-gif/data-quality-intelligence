"""Profiles a DataFrame: per-column statistics used as the basis for scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    """Summary statistics for a single column."""

    name: str
    dtype: str
    count: int
    missing: int
    missing_pct: float
    unique: int
    unique_pct: float
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    top_values: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "count": self.count,
            "missing": self.missing,
            "missing_pct": round(self.missing_pct, 4),
            "unique": self.unique,
            "unique_pct": round(self.unique_pct, 4),
            "is_numeric": self.is_numeric,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "top_values": self.top_values,
        }


@dataclass
class DatasetProfile:
    """Profile for an entire DataFrame: row/column counts plus per-column profiles."""

    row_count: int
    column_count: int
    columns: Dict[str, ColumnProfile]
    duplicate_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "duplicate_rows": self.duplicate_rows,
            "columns": {name: col.to_dict() for name, col in self.columns.items()},
        }


class DataProfiler:
    """Computes descriptive statistics for a pandas DataFrame.

    Example:
        profile = DataProfiler(df, top_n=5).profile()
        print(profile.columns["age"].missing_pct)
    """

    def __init__(self, df: pd.DataFrame, top_n: int = 5):
        self.df = df
        self.top_n = top_n

    def profile(self) -> DatasetProfile:
        columns: Dict[str, ColumnProfile] = {}
        for col_name in self.df.columns:
            columns[col_name] = self._profile_column(col_name)

        return DatasetProfile(
            row_count=len(self.df),
            column_count=len(self.df.columns),
            columns=columns,
            duplicate_rows=int(self.df.duplicated().sum()),
        )

    def _profile_column(self, col_name: str) -> ColumnProfile:
        series = self.df[col_name]
        count = len(series)
        missing = int(series.isna().sum())
        missing_pct = (missing / count) if count else 0.0
        unique = int(series.nunique(dropna=True))
        unique_pct = (unique / count) if count else 0.0
        is_numeric = pd.api.types.is_numeric_dtype(series)

        stats = {}
        if is_numeric:
            non_null = series.dropna()
            if len(non_null):
                stats = {
                    "min": float(non_null.min()),
                    "max": float(non_null.max()),
                    "mean": float(non_null.mean()),
                    "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
                    "median": float(non_null.median()),
                }

        top_values = (
            series.value_counts(dropna=True).head(self.top_n).index.tolist()
        )

        return ColumnProfile(
            name=col_name,
            dtype=str(series.dtype),
            count=count,
            missing=missing,
            missing_pct=missing_pct,
            unique=unique,
            unique_pct=unique_pct,
            is_numeric=is_numeric,
            top_values=top_values,
            **stats,
        )
