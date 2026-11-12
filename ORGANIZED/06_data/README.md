# 06_data: Data Catalog & References

This directory documents the data sources used throughout the project. Per project guidelines, raw and inference data files are **not duplicated** here to preserve disk space and maintain a single source of truth.

---

## 1. Primary Data Sources

Both datasets reside at the project root:

| Dataset | Relative Path | Size | Description |
|---|---|---|---|
| **Training History** | `../../history.csv` | ~3.42 MB | Historical telecom customer churn data used for exploratory analysis, feature store engineering, imputer training, and churn model training (5,634 rows ingested / 5,912 raw). |
| **Inference Set** | `../../inference.csv` | ~852 KB | Unlabeled customer cohort for production scoring (1,409 rows ingested / 1,461 raw). Contains leaked textual column `CustomerFeedback` which is strictly excluded from scoring features. |

---

## 2. Ingestion & Storage Architecture in Databricks

In the Databricks production environment, these CSVs are ingested into Delta format under Unity Catalog:
- `workspace.default.train` (from `history.csv`)
- `workspace.default.inference` (from `inference.csv`)

### Derived Tables & Feature Stores:
- `workspace.default.train_silver` (ETL hygiene, standardization)
- `workspace.default.feature_store_train` & `workspace.default.feature_store_inference`
- `workspace.default.target_store_train` & `workspace.default.target_store_inference`
- `workspace.default.abt_1` (Regression ABT for `TotalCharges`)
- `workspace.default.abt_2` (Classification ABT for `Churn`)
- `workspace.default.abt_inference` (Inference scoring ABT)

---

## 3. Data Hygiene & Known Peculiarities

1. **`tenure` Anomalies**:
   - Contains string `'unknown'` values (19 instances in history).
   - In SQL/Databricks: cast via `TRY_CAST(tenure AS BIGINT)`.
   - In Bruno's exploration: investigated via `TotalCharges / MonthlyCharges`.
2. **`TotalCharges` Missingness**:
   - Missing in customers where `tenure == 0` or blank strings.
   - Raquel's pipeline trains a Ridge regression imputer registered as `@champion` in MLflow to predict missing values.
   - Bruno's notebook imputes deterministically via `MonthlyCharges * tenure` (and fills `tenure == 0` with `MonthlyCharges`).
3. **`CustomerFeedback` Target Leakage**:
   - Present in `inference.csv` and contains direct statements describing customer outcome/cancellation.
   - **Mandatory Decision**: Strictly excluded from all churn classification training and inference feature sets.
