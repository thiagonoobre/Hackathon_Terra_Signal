# Pipeline Notebooks Description

## 1. Pipeline Overview
1. [01_EDA_and_data_audit (train).ipynb](./../02_etl_and_features/notebooks/01_eda_data_audit_train.ipynb): Profiles raw training data, audits distributions, evaluates missingness, and analyzes decile churn rates.
2. [02_EDA_and_data_audit (inference).ipynb](./../02_etl_and_features/notebooks/02_eda_data_audit_inference.ipynb): Profiles unlabelled inference data to detect covariate drift and structural schema alignment against training data.
3. [03_ETL_sql.ipynb](./../02_etl_and_features/notebooks/03_etl_sql.ipynb): Cleans data in Databricks SQL, standardizes categorical encodings, constructs `train_silver`, and formulates `TotalCharges` imputation strategies.
4. [04_Feature_Store.ipynb](./../02_etl_and_features/notebooks/04_feature_store.ipynb): Executes external SQL feature pipelines to build Delta feature and target store tables in Unity Catalog.
5. [05_TRAIN1_TotalCharges.ipynb](./../03_modeling/01_totalcharges_imputer/06.0_train1_totalcharges.ipynb): Assembles ABT 1, builds correlation/VIF screens, trains and logs a Ridge regressor via MLflow, and registers it as `@champion` for `TotalCharges` imputation.
6. [06_TRAIN2_Churn.ipynb](./../03_modeling/02_churn_classification/08.0_train2_churn.ipynb): Assembles ABT 2, pulls the champion imputer from MLflow, trains and tunes 7 candidate classification models, and logs results to Unity Catalog.

---

## 2. Index Table

| # | Notebook | Purpose | Inputs | Outputs | Libs | Runtime | Status |
|---|---|---|---|---|---|---|---|
| 01 | [01_EDA_and_data_audit (train).ipynb](../02_etl_and_features/notebooks/01_eda_data_audit_train.ipynb) | Profile training features, audit data quality, and compute decile churn rates | `workspace.default.train` | Visual EDA plots, summary stats | `pandas`, `numpy`, `seaborn`, `matplotlib` | ~1-2 min | Complete |
| 02 | [02_EDA_and_data_audit (inference).ipynb](../02_etl_and_features/notebooks/02_eda_data_audit_inference.ipynb) | Audit inference dataset structure and check distribution shifts | `workspace.default.inference` | Visual EDA plots, missingness tables | `pandas`, `numpy`, `seaborn`, `matplotlib` | ~1 min | Complete |
| 03 | [03_ETL_sql.ipynb](../02_etl_and_features/notebooks/03_etl_sql.ipynb) | SQL transformations, data hygiene, customer overlap checks, and feature drafting | `workspace.default.train`, `workspace.default.inference` | `workspace.default.train_silver` | Spark SQL | ~30-60 sec | Complete |
| 04 | [04_Feature_Store.ipynb](../02_etl_and_features/notebooks/04_feature_store.ipynb) | Materialize curated Delta Feature Store and Target Store tables | `feature_store.sql`, `target_store.sql`, `target_store_inference.sql`, raw tables | `feature_store_{train\|inference}`, `target_store_{train\|inference}` | `pyspark`, `pandas` | ~1-2 min | Complete |
| 05 | [05_ABT1_TotalCharges.ipynb](../03_modeling/01_totalcharges_imputer/05.0_abt1_totalcharges.ipynb) | Train, evaluate, and register champion `TotalCharges` regression imputer | `feature_store_train`, `target_store_train` | `abt_1`, MLflow model `total_charges_ridge_tuned@champion` | `sklearn`, `mlflow`, `pandas`, `seaborn` | ~2-3 min | Complete |
| 06 | [06_ABT2_Churn.ipynb](../03_modeling/02_churn_classification/07.0_abt2_churn.ipynb) | Build classification ABT, load imputer, and benchmark 7 churn ML models | `feature_store_train`, `target_store_train`, champion imputer model | MLflow experiment runs (LogReg, RF, LGBM, XGBoost, MLP, DT, KNN) | `sklearn`, `lightgbm`, `xgboost`, `feature_engine`, `mlflow` | ~5-8 min | Complete |

---

## 3. Per-Notebook Breakdown

### 01_EDA_and_data_audit (train).ipynb
- **Purpose**: Exploratory analysis and audit of training dataset.
- **Inputs**: Unity Catalog table `workspace.default.train`.
- **Outputs**: Univariate distribution plots, frequency charts, and decile log-odds churn diagnostics.
- **Key Steps**: Load PySpark table to pandas; compute missingness; parse categorical and numerical features; analyze monotonic relationships across charge deciles.
- **Dependencies**: Unity Catalog table `train`; Seaborn library.
- **Caveats**: Converts entire dataset into single-node Pandas DataFrame in memory (`toPandas()`); column `tenure` has 'unknown' values stored as string.

### 02_EDA_and_data_audit (inference).ipynb
- **Purpose**: Data quality audit and distribution validation on unlabeled inference set.
- **Inputs**: Unity Catalog table `workspace.default.inference`.
- **Outputs**: Feature distributions and missingness tables for inference data.
- **Key Steps**: Mirror training EDA functions; evaluate missingness counts; audit categorical categories and numeric ranges; verify absence of target `Churn`.
- **Dependencies**: Unity Catalog table `inference`; Seaborn library.
- **Caveats**: `Churn` decile analysis cells are commented out due to lack of labels in inference; inherits identical tenure and categorical anomalies.

### 03_ETL_sql.ipynb
- **Purpose**: SQL-based validation, schema cleaning, and feature engineering logic prototyping.
- **Inputs**: `workspace.default.train`, `workspace.default.inference`.
- **Outputs**: Managed Delta table `workspace.default.train_silver`.
- **Key Steps**: Check row count discrepancies (train 5634 vs 5912; inference 1409 vs 1461); verify zero customer ID overlap; build silver table with cleaned tenure and contract aggregations; evaluate median charge groupings.
- **Dependencies**: Databricks SQL engine; raw tables `train` and `inference`.
- **Caveats**: Hardcodes `workspace.default` schema; noted ingestion drop of rows between source CSVs and catalog tables.

### 04_Feature_Store.ipynb
- **Purpose**: Batch execution of parameterized SQL scripts to create Delta Feature Store and Target Store tables.
- **Inputs**: SQL files `../02_etl_and_features/sql/feature_store.sql`, `../02_etl_and_features/sql/target_store.sql`, `../02_etl_and_features/sql/target_store_inference.sql`, and raw Unity Catalog tables.
- **Outputs**: Delta tables `feature_store_train`, `feature_store_inference`, `target_store_train`, `target_store_inference`.
- **Key Steps**: Loop across datasets dictionary; read relative SQL template queries; execute string formatting on `{dataset}`; overwrite Delta tables in Unity Catalog.
- **Dependencies**: Pre-existing SQL scripts in `../02_etl_and_features/sql/`; Spark runtime.
- **Caveats**: Uses relative file paths `../02_etl_and_features/sql/` which require fixed working directory execution; relies on table overwrite mode.

### 05_ABT1_TotalCharges.ipynb
- **Purpose**: Feature selection, model training, and registration of `TotalCharges` imputation regression pipeline.
- **Inputs**: `workspace.default.feature_store_train`, `workspace.default.target_store_train`.
- **Outputs**: Table `workspace.default.abt_1`; registered Unity Catalog model `workspace.default.total_charges_ridge_tuned@champion`.
- **Key Steps**: Join feature and target stores; compute Pearson correlation matrix and prune collinear features; define `train_and_log` harness with GridSearchCV; benchmark OLS vs Ridge; set champion alias in MLflow.
- **Dependencies**: `04_Feature_Store.ipynb` artifacts; MLflow Tracking & Model Registry (ID: 1316272881415465).
- **Caveats**: Target leakage variables (`TotalSpend`, `TotalSpendPct`) must be manually excluded; cell 35 has commented-out model reloading check.

### 06_ABT2_Churn.ipynb
- **Purpose**: End-to-end churn classification modeling pipeline, hyperparameter tuning, and model tracking.
- **Inputs**: `feature_store_train`, `target_store_train`, and loaded model `models:/workspace.default.total_charges_ridge_tuned@champion`.
- **Outputs**: Registered classification models across 7 algorithm families in MLflow (Experiment ID: 1203535636207065).
- **Key Steps**: Merge feature store; invoke champion regression model to impute missing `TotalCharges`; configure feature preprocessors (mean, median, binary, one-hot); train and tune Logistic Regression, Random Forest, LightGBM, XGBoost, MLP, Decision Tree, and KNN.
- **Dependencies**: `05_ABT1_TotalCharges.ipynb` champion imputer model in Unity Catalog; `feature_engine`, `lightgbm`, `xgboost`.
- **Caveats**: Notebook execution finishes before writing the final comparison markdown summary cell; inference scoring pipeline is not automated inside this notebook.

---

## 4. Shared Conventions
- **Catalog & Storage**: Uniform namespace `workspace.default.*` inside Unity Catalog using Delta Lake format.
- **Data Exchange**: Spark tables converted to local pandas DataFrames (`toPandas()`) for modeling workloads.
- **Tracking & Governance**: Centralized MLflow Tracking with explicit experiment IDs and Unity Catalog model naming conventions (`workspace.default.<model_name>`).
- **Pipeline Architecture**: Dual-ABT approach separating missing-value imputation (ABT 1) from final churn classification (ABT 2).
- **Coding Style**: Reusable functional training wrappers (`train_and_log`, `train_and_log_churn`) coupling scikit-learn Pipeline with MLflow autologging.

---

## 5. Known Gaps
- **Row Count Inconsistencies**: Ingestion discrepancy identified in notebook 03 (train count is 5634 vs expected 5912; inference count is 1409 vs expected 1461).
- **Inference Scoring Pipeline**: No standalone end-to-end batch inference notebook generating final predictions table on `feature_store_inference`.
- **Hardcoded Catalog Path**: Schema namespace `workspace.default` and MLflow Experiment IDs are hardcoded, limiting cross-environment migration (dev/staging/prod).
- **Relative SQL Pathing**: `04_Feature_Store.ipynb` relies on `../02_etl_and_features/sql/` relative filesystem path rather than Databricks Workspace Repos / package structures.
- **Missing Final Model Selection Cell**: Cell 24 in `06_ABT2_Churn.ipynb` leaves the champion model selection markdown unpopulated.
