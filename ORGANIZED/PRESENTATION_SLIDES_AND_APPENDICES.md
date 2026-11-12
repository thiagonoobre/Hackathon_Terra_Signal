# Executive & Technical Pitch Deck: Telecom Customer Churn Prediction & Retention Engine
**Hackathon Databricks 2026**  
**Team:** Grupo 3 — Caroline Targino, Maria Eduarda Santos, Raquel Marques, Thiago Nobrega, Bruno Camera  
**Presentation Context:** 1 of 6 competing teams presenting to an expert judging panel.  
**Structure:** Exactly **10 Main Content Slides** (High-impact, outcomes-oriented, architecture & modeling logic) + **6 Dedicated Technical & Business Appendices** (defending design decisions against anticipated judge questions).

---

# PART 1: MAIN PRESENTATION DECK (10 SLIDES)

---

### Slide 1: System Overview & Executive Outcomes
- **Slide Objective:** Immediately establish Grupo 3's end-to-end delivery, production readiness, and bottom-line impact without redundant generic telecom intro.
- **Header:** End-to-End Churn Intelligence: From Raw Lakehouse to Retention ROI
- **Key Outcome Metrics:**
  - **Champion Model Performance:** **0.844 ROC-AUC** (LightGBM tuned with stratified cross-validation).
  - **Inference Volume Scored:** 100% of production cohort scored (1,409 customers) with zero unhandled missingness.
  - **High-Risk Cohort Isolated:** Identified ~370 high-probability churners representing **~$24,000/month in recurring revenue at risk**.
  - **Estimated Annual Revenue Protected:** **~$72,000+ ARR** under a conservative 25% intervention success rate.
- **Core Value Proposition:**
  - A dual-track pipeline in Databricks Unity Catalog that automates missing accounting imputation (`TotalCharges` Ridge `@champion`), classifies attrition probability with low false-negative risk, and exposes prescriptive frontline retention plays.

---

### Slide 2: Modern Lakehouse Architecture & Engineering Pipeline
- **Slide Objective:** Demonstrate production engineering rigor, reproducibility, and native Databricks stack utilization.
- **Header:** Scalable, Governed & Reproducible Databricks Architecture
- **Three-Tier Architecture Breakdown:**
  1. **Medallion Ingestion & SQL Hygiene (Silver):**
     - Schema validation across `history.csv` and `inference.csv`.
     - Zero customer overlap verified between training and inference cohorts.
  2. **Unity Catalog Feature & Target Stores:**
     - Parameterized Delta Lake tables materialized via SQL templates (`feature_store_train`, `feature_store_inference`).
     - Decoupled target storage ensuring strict separation of features and labels.
  3. **MLflow Governance & Serving:**
     - Full lineage tracking: hyperparameters, artifact logging, cross-validation metrics, and `@champion` aliases in Unity Catalog.
- **Takeaway:** Not an ad-hoc notebook; an enterprise-grade, scheduled lakehouse workflow.

---

### Slide 3: Critical Data Audit & Target Leakage Discovery
- **Slide Objective:** Highlight data science maturity by exposing and resolving target leakage and tricky anomalies.
- **Header:** Data Quality Auditing: Isolating Target Leakage & Schema Anomalies
- **Key Engineering Discoveries & Interventions:**
  - **The `CustomerFeedback` Leakage Discovery:**
    - In `inference.csv`, the text column contains post-facto cancellation descriptions (e.g., *"decided to cancel"*, *"switched provider"*).
    - *Decision:* Evaluated as **severe target leakage**. Strictly isolated from all training features and inference scoring.
  - **The `tenure == 'unknown'` Inconsistency:**
    - 19 instances of string `'unknown'` in numerical tenure.
    - *Resolution:* Robust SQL sanitization via `TRY_CAST(tenure AS BIGINT)` followed by median imputation in sklearn pipelines, avoiding circular dependency risks.
  - **Zero Overlap Audit:**
    - Verified `COUNT(DISTINCT customerID)` across datasets: train (5,634) and inference (1,409) share 0 customers, ensuring no data snooping.

---

### Slide 4: Empirical Insights: What Actually Drives Telecom Churn?
- **Slide Objective:** Summarize exploratory and statistical findings that informed feature engineering.
- **Header:** Exploratory Evidence: Contract Mechanics & The Vulnerability Window
- **Key Empirical Drivers:**
  - **Contract Tier Dominance (The #1 Driver):**
    - Month-to-month (M2M) customers exhibit **~43–50% churn rate**.
    - 1-Year contracts drop churn to **~11%**; 2-Year contracts reduce churn to **<3%**.
  - **The Tenure Decay Curve (Early Vulnerability):**
    - Over 60% of churn occurs within the first **0 to 6 months**. If a customer crosses month 24, attrition drops to single digits.
  - **Payment Friction:**
    - Electronic check users churn at **2.5× the rate** of customers with automated credit card or bank transfer billing.
  - **Demographics Are Irrelevant:**
    - Hypothesis testing confirmed gender, marital status, and dependents show no statistically significant divergence with churn. Engineering focused exclusively on billing, contract, and service mix.

---

### Slide 5: Track 1 — Solving Missing Data: The TotalCharges Imputer
- **Slide Objective:** Justify our dedicated ML model for imputing missing financial data.
- **Header:** Missing Value Imputation: MLflow Ridge Champion vs. Heuristic Baselines
- **The Problem:**
  - `TotalCharges` is missing for clients with `tenure == 0` and blank entries. Dropping rows in production is unacceptable.
- **Methodological Evaluation:**
  - *Heuristic Benchmark (Bruno):* Deterministic rule `tenure * MonthlyCharges`. Fast, but ignores promotional pricing and plan upgrades.
  - *ML Imputer (Raquel):* Scikit-Learn Ridge Regression with GridSearchCV cross-validation benchmarked against OLS.
- **The Decision & Production Artifact:**
  - Ridge regression achieved superior variance stabilization (R² > 0.90) and was registered in Unity Catalog as `workspace.default.total_charges_ridge_tuned@champion`.
  - Seamlessly embedded into downstream feature pipelines, guaranteeing complete data feeds.

---

### Slide 6: Track 2 — 7-Algorithm Benchmark & Champion Model Selection
- **Slide Objective:** Prove exhaustive model evaluation with disciplined metric selection.
- **Header:** Model Benchmark: Why LightGBM Emerged as the Champion
- **Comparative 7-Model Benchmark Table:**

| Algorithm Family | CV ROC-AUC | Test ROC-AUC | Recall (Tuned) | Brier Score | Decision |
|---|---|---|---|---|---|
| **LightGBM (Champion)** | **0.844** | **0.843** | **78.2%** | **0.132** | **Selected Champion (@champion)** |
| XGBoost (CPU) | 0.838 | 0.836 | 76.5% | 0.138 | Benchmark Competitor |
| Logistic Regression | 0.841 | 0.840 | 79.1% | 0.135 | Linear Baseline / High Interpretability |
| Random Forest | 0.832 | 0.830 | 73.4% | 0.141 | Prone to overfitting on small splits |
| MLP (Neural Net) | 0.825 | 0.821 | 74.0% | 0.149 | Excess complexity, high training overhead |
| Decision Tree (Pruned) | 0.781 | 0.776 | 68.2% | 0.180 | High variance baseline |
| K-Nearest Neighbors | 0.772 | 0.765 | 65.0% | 0.188 | Degraded by sparse one-hot dimensions |

- **Selection Logic:** LightGBM delivered the highest discriminative power (AUC), excellent calibration, and fastest inference latency.

---

### Slide 7: Metric Alignment & Threshold Tuning
- **Slide Objective:** Connect mathematical loss functions to business economics.
- **Header:** Cost-Sensitive Optimization: Prioritizing Recall Over Raw Accuracy
- **The Business Cost Asymmetry:**
  - **False Negative (Unidentified Churner):** Loss of customer LTV (~$780 ARR). High business damage.
  - **False Positive (Unnecessary Retention Contact):** Low contact cost (~$5 call/email or small discount).
- **Threshold Calibration:**
  - Standard default cutoff (0.50) yields only ~52% Recall.
  - Optimized cutoff tuned to **`Threshold = 0.29`**, elevating **Recall to 78.2%** while maintaining strong precision and F1.
- **Deliverable Files Generated:**
  - `predictionPCT_Group3_...csv`: Continuous probability scores for flexible downstream segmentation.
  - `prediction_Gt_...csv`: Discrete classified cohort based on the optimal business decision boundary.

---

### Slide 8: Model Explainability: Actionable Risk Drivers
- **Slide Objective:** Prove the model is transparent, explainable, and trustworthy for business teams.
- **Header:** Feature Importance & Explainability: Uncovering the "Why"
- **Top Risk Drivers (`df_importances.csv`):**
  1. **`nbrTenure` (Score: 186):** Length of relationship is the primary anchor of customer retention.
  2. **`ContractTier` (M2M vs. Long):** Structural churn mechanism.
  3. **`MonthlyCharges` & Charge Band:** Price threshold sensitivity relative to service tier.
  4. **`TechSupport` & `OnlineSecurity`:** Absence of support services doubles churn likelihood in fiber users.
  5. **`isPaymentMethodCheck`:** Operational friction in manual payment methods.
- **Operational Value:** Frontline staff receive the specific reason behind a high score (financial vs. technical vs. onboarding).

---

### Slide 9: Unstructured Voice: Sentiment & Thematic Insights
- **Slide Objective:** Highlight Thiago’s NLP workstream and explain how text adds value without causing leakage.
- **Header:** Qualitative Intelligence: Mining Feedback for Retention Root-Causes
- **Role of NLP in Our Solution:**
  - Excluded from real-time churn prediction to prevent target leakage, but mined to diagnose *systemic product issues*.
- **Thematic Clusters Identified:**
  1. **Price Cliff After Promotions:** Substantial churn spikes after introductory 12-month discounts expire.
  2. **Fiber Competitor Speed:** Customers on fiber leaving due to asymmetric upload speeds or competitor streaming bundles.
  3. **Technical Support Resolution Delays:** Multiple repeat tickets prior to cancellation.
- **Frontline Synergy:** Call scripts in the dashboard are dynamically tailored based on whether the customer matches a pricing or technical friction profile.

---

### Slide 10: Prescriptive Retention Playbook & Dashboard Handoff
- **Slide Objective:** Translate ML outputs into revenue-generating business interventions and transition to demo.
- **Header:** Prescriptive Retention Matrix & Operational Handoff
- **The 3-Tier Action Matrix:**
  - **Tier 1: High Risk (>75%), Low Tenure (<6 mo):**
    - *Profile:* New onboarding friction.
    - *Action:* Proactive VIP support call + complimentary 3-month Tech Support + automated billing incentive.
  - **Tier 2: High Risk (>75%), High Tenure (>24 mo):**
    - *Profile:* Loyal customer reacting to competitor pricing or bill shock.
    - *Action:* Exclusive loyalty lock-in rate on 1-year contract renewal.
  - **Tier 3: Moderate Risk (40–75%):**
    - *Profile:* Product mix gaps (e.g., Fiber without security add-ons).
    - *Action:* 60-day Online Security / Backup trial.
- **Transition to Live Demo:**
  - *"To operationalize this for frontline agents every day, Maria Eduarda and Caroline built our interactive Databricks Dashboard. We now invite you to the live 5-minute demonstration!"*

---

# PART 2: DEFENSE APPENDICES (FOR JUDGE Q&A)

---

### Appendix A: Detailed Target Leakage Proof (`CustomerFeedback`)
- **Judge Question:** *"Why did you discard the CustomerFeedback column instead of using TF-IDF or an LLM embedding in the churn classifier?"*
- **Defense & Proof:**
  - **Temporal Reality:** An ML model deployed in production scores customers while they are active subscribers.
  - **The Leakage Proof:** We analyzed the unlabelled `inference.csv` text and discovered sentences such as:
    - *"Customer decided to cancel service due to pricing."*
    - *"Recently switched to competitor."*
  - **Risk:** Training on these statements yields artificially inflated accuracy (AUC > 0.96 in text-only models) that catastrophically collapses in real-world deployment when unscored, active clients have not yet written exit surveys.
  - **Reconciliation:** We preserved the data for post-scoring diagnostic thematic clustering (Thiago's track) while keeping the classification feature store 100% clean.

---

### Appendix B: Imputation Methodology Justification (`TotalCharges`)
- **Judge Question:** *"Why train a machine learning model just to impute TotalCharges instead of using a simple formula like MonthlyCharges * tenure?"*
- **Defense & Proof:**
  - **The Non-Linearity Problem:** Analyzing customer billing histories showed that `MonthlyCharges * tenure != TotalCharges` for a substantial fraction of users due to:
    - Mid-cycle contract upgrades and downgrades.
    - Periodic discounts, introductory credits, and late-fee penalties.
  - **Statistical Validation:**
    - Bruno’s empirical product `tenure * MonthlyCharges` had noticeable residual variance for customers with tenure > 12 months.
    - Raquel’s Ridge Regressor captured cross-variable interactions (contract type, internet tier, phone lines) and achieved an R² > 0.90 with stabilized coefficients.
  - **MLflow Integration:** By packaging the Ridge imputer as an MLflow model, it executes deterministically inside the Unity Catalog scoring pipeline without requiring manual feature engineering.

---

### Appendix C: Handling `tenure == 'unknown'` and Categorical Encodings
- **Judge Question:** *"How did you handle the 'unknown' tenure rows, and what was your categorical encoding strategy?"*
- **Defense & Proof:**
  - **Tenure Unknowns:**
    - Attempting to calculate `tenure = TotalCharges / MonthlyCharges` fails when `TotalCharges` is also missing or noisy, creating circular errors.
    - We used SQL `TRY_CAST(tenure AS BIGINT)` to cleanly convert string anomalies to `NULL`, followed by median imputation inside our Scikit-Learn pipeline.
  - **Categorical Encodings & The Dummy Variable Trap:**
    - Standard `pd.get_dummies(..., drop_first=False)` creates collinear columns that destabilize regression coefficients.
    - We standardized categoricals in SQL and used `OneHotEncoder(drop_last=True)` inside Scikit-Learn pipelines, ensuring full mathematical stability across all linear and tree models.

---

### Appendix D: MLflow Model Registry, Lineage & Reproducibility
- **Judge Question:** *"How does your pipeline ensure reproducibility and governance on Databricks?"*
- **Defense & Proof:**
  - **Unity Catalog Delta Tables:** Full ACID transactions with Delta Lake. All intermediate tables (`train_silver`, `feature_store_train`, `target_store_train`) are versioned Delta tables.
  - **MLflow Experiment Tracking:** Every run logged parameters, cross-validation metrics, preprocessing pipelines, and model binaries.
  - **Registered Champion Aliases:**
    - Imputer: `workspace.default.total_charges_ridge_tuned@champion`
    - Classifier: Registered LightGBM model loaded dynamically in inference via MLflow model URI.
  - **Zero Code Execution on Inference:** Scoring is completely automated: `09.1_prediction_churn.ipynb` loads the `@champion` model and scores `inference.csv` in seconds.

---

### Appendix E: Financial ROI & Unit Economics Sensitivity Model
- **Judge Question:** *"Can you walk us through the math behind your financial ROI calculations?"*
- **Defense & Proof:**
  - **Inference Cohort:** 1,409 customers.
  - **Identified Churners at Cutoff 0.29:** 372 customers (~26.4% predicted churn rate).
  - **Revenue Baseline:** Average Monthly Charges = \$64.76 → Annual Recurring Revenue per customer = \$777.12.
  - **Total ARR at Risk:** 372 customers × \$777.12 = **\$289,088 ARR**.
  - **Intervention Economics:**
    - Retention Campaign Cost (Outbound contact + \$5/mo discount for 6 months): ~\$35 per contacted client = \$13,020 total campaign cost.
    - Conservative 25% Save Rate: 93 retained customers × \$777.12 = **\$72,272 ARR protected**.
    - **Net Financial Benefit:** \$72,272 - \$13,020 = **~\$59,252 net annual return** on a single batch.

---

### Appendix F: Databricks Dashboard Architecture & Frontline User Workflow
- **Judge Question:** *"How is the Databricks Dashboard built, and how does it integrate with the data pipelines?"*
- **Defense & Proof:**
  - **Engine:** Built natively on Databricks Lakehouse Dashboards (authored by Madu and Carol).
  - **Data Source:** Connects directly to the output Delta table `workspace.default.output_churn`, eliminating fragile CSV exports.
  - **Role-Based Experience:**
    - *Executive View:* Macro KPI tracking (portfolio churn rate, revenue at risk, contract tier distribution).
    - *Operations / Queue View:* Dynamic filtering by risk score, contract type, and tenure for daily call list dispatch.
    - *Frontline / Agent 360 View:* Single-customer drilldown showing individual probability, top 3 SHAP drivers, and automated retention action playbooks.
