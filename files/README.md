# Data Quality Intelligence (DQI)

A lightweight Python framework for assessing the quality of tabular data:
profile it, validate it against configurable rules, flag statistical
anomalies, and roll everything up into a quality score and report.

Built on `pandas` / `numpy`.

## Core Concepts

| Component         | Role                                                                    |
|---------------------|----------------------------------------------------------------------------|
| `DataProfiler`      | Computes per-column stats: missing %, uniqueness, min/max/mean, top values. |
| `Rule` / `RuleEngine` | Declarative validation rules (not-null, unique, range, regex, allowed values, custom) run against columns. |
| `AnomalyDetector`   | Flags statistical outliers in numeric columns (IQR or z-score).           |
| `QualityScorer`     | Combines completeness + rule pass-rates into per-column and overall scores (A–F grade). |
| `ReportGenerator`   | Renders everything as Markdown or JSON.                                    |

```
DataFrame ──► DataProfiler ──► profile ─┐
         ├──► RuleEngine ─────► rule results ─┼──► QualityScorer ──► score ──► ReportGenerator ──► report
         └──► AnomalyDetector ─► anomalies ────┘
```

## Project Structure

```
data_quality_intelligence/
├── dqi/
│   ├── __init__.py     # public API exports
│   ├── profiler.py       # DataProfiler, ColumnProfile, DatasetProfile
│   ├── rules.py            # Rule base + built-ins, RuleEngine
│   ├── anomaly.py          # AnomalyDetector (IQR / z-score)
│   ├── scoring.py          # QualityScorer, QualityScore, ColumnScore
│   └── report.py            # ReportGenerator (Markdown / JSON)
├── examples/
│   └── example_run.py        # runnable demo with intentionally messy data
├── tests/
│   └── test_dqi.py             # pytest unit tests (14 tests)
├── requirements.txt
└── README.md
```

## Quick Start

```python
import pandas as pd
from dqi import (
    DataProfiler, RuleEngine, NotNullRule, UniqueRule, RangeRule,
    AnomalyDetector, QualityScorer, ReportGenerator,
)

df = pd.read_csv("customers.csv")

# 1. Profile
profile = DataProfiler(df).profile()

# 2. Validate against rules
engine = RuleEngine([
    NotNullRule("customer_id"),
    UniqueRule("customer_id"),
    RangeRule("age", min_value=0, max_value=120),
])
rule_results = engine.run(df)

# 3. Detect statistical anomalies
anomalies = AnomalyDetector(df, method="iqr").detect(columns=["age", "income"])

# 4. Score
score = QualityScorer().score(profile, rule_results)
print(f"Overall: {score.overall:.1%} (Grade {score.grade})")

# 5. Report
report = ReportGenerator(profile, rule_results, score, anomalies, dataset_name="customers.csv")
print(report.to_markdown())
report.save("quality_report.md")
```

Run the bundled demo (a deliberately messy dataset — duplicate IDs, bad
emails, an out-of-range age, a disallowed country code, and an income
outlier):

```bash
python examples/example_run.py
```

## Built-in Rules

| Rule                 | Fails when...                                      |
|------------------------|-------------------------------------------------------|
| `NotNullRule`          | value is missing (NaN/None)                            |
| `UniqueRule`             | value is a duplicate (first occurrence passes)          |
| `RangeRule`               | numeric value is outside `[min_value, max_value]`         |
| `RegexRule`                 | string value doesn't match the given pattern                |
| `AllowedValuesRule`           | value isn't in the allowed set                                |
| `TypeRule`                      | value isn't an instance of the expected type                    |
| `CustomRule`                      | a user-supplied `value -> bool` predicate returns False           |

All rules skip missing values (that's `NotNullRule`'s job) except `NotNullRule` itself, so you can combine rules without double-penalizing a single missing cell.

## Scoring Model

Per-column score = `completeness_weight * (1 - missing_pct) + validity_weight * (rule pass rate)`,
default weights 0.5/0.5. Uniqueness is reported but not scored by default,
since high/low uniqueness isn't inherently good or bad (IDs vs. categorical
columns have very different expectations). Overall score is the mean of all
column scores, mapped to a letter grade:

| Score    | Grade |
|-----------|-------|
| ≥ 95%     | A     |
| ≥ 85%     | B     |
| ≥ 70%     | C     |
| ≥ 50%     | D     |
| < 50%     | F     |

Adjust weights via `QualityScorer(completeness_weight=..., validity_weight=...)`.

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Extending

- **New rule**: subclass `Rule` and implement `_is_valid(value)`, or wrap a
  function with `CustomRule(column, predicate, name="...")`.
- **New anomaly method**: extend `AnomalyDetector._detect_column` or swap in
  a different detector (e.g. Isolation Forest) that returns `AnomalyResult`
  objects with the same shape.
- **Custom scoring**: subclass or reconfigure `QualityScorer`, or write your
  own aggregation over `DatasetProfile` + `RuleResult` — both are plain
  dataclasses with `.to_dict()` for easy serialization.
- **Output formats**: `ReportGenerator.to_dict()` gives you a plain dict if
  you want to render HTML, push to a dashboard, or write to a database.
