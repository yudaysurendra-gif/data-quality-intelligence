"""
End-to-end example: build a deliberately messy customer dataset, profile it,
validate it against rules, detect anomalies, score it, and print a report.

Run with:
    python examples/example_run.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dqi import (
    DataProfiler,
    RuleEngine,
    NotNullRule,
    UniqueRule,
    RangeRule,
    RegexRule,
    AllowedValuesRule,
    AnomalyDetector,
    QualityScorer,
    ReportGenerator,
)


def build_sample_data() -> pd.DataFrame:
    """A small, intentionally messy dataset: missing values, dupes, bad emails,
    an out-of-range age, and one wild income outlier."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 5, 7, 8],  # duplicate id (5)
            "email": [
                "alice@example.com",
                "bob@example.com",
                "not-an-email",
                "dana@example.com",
                None,  # missing
                "erin@example.com",
                "frank@example.com",
                "grace@example.com",
            ],
            "age": [34, 29, 41, 150, 25, 38, -5, 30],  # 150 and -5 are invalid
            "country": ["US", "US", "CA", "UK", "US", "US", "MX", "ZZ"],  # ZZ not allowed
            "income": [55000, 61000, 58000, 62000, 59500, 60000, 57000, 5_000_000],  # outlier
        }
    )


def main() -> None:
    df = build_sample_data()

    # --- Profiling ------------------------------------------------------
    profile = DataProfiler(df).profile()

    # --- Rule validation --------------------------------------------------
    engine = RuleEngine(
        [
            NotNullRule("customer_id"),
            UniqueRule("customer_id"),
            NotNullRule("email"),
            RegexRule("email", pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
            RangeRule("age", min_value=0, max_value=120),
            AllowedValuesRule("country", allowed=["US", "CA", "UK", "MX"]),
        ]
    )
    rule_results = engine.run(df)

    # --- Anomaly detection --------------------------------------------------
    anomaly_results = AnomalyDetector(df, method="iqr").detect(columns=["age", "income"])

    # --- Scoring ------------------------------------------------------------
    score = QualityScorer().score(profile, rule_results)

    # --- Report --------------------------------------------------------------
    report = ReportGenerator(
        profile=profile,
        rule_results=rule_results,
        score=score,
        anomaly_results=anomaly_results,
        dataset_name="customers.csv",
    )

    print(report.to_markdown())


if __name__ == "__main__":
    main()
