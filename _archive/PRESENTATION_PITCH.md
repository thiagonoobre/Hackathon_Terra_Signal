# Pitch Presentation: End-to-End Churn Intelligence & Retention System
**Event:** Databricks Hackathon 2026  
**Team:** Grupo 3 — Caroline Targino, Maria Eduarda Santos, Raquel Marques, Thiago Nobrega, Bruno Camera  
**Format:** 15-Minute Pitch (10 Minutes Executive & Technical Presentation + 5 Minutes Live Dashboard Demo)

---

## Slide Timing & Structure Summary

| Slide # | Slide Title | Theme & Focus | Time Allocated |
|---|---|---|---|
| **Slide 1** | Executive Summary & Business Problem | Value Proposition, Churn Impact & Opportunity | 1:00 min |
| **Slide 2** | Architecture & Modern Data Stack | Unity Catalog, Delta Lake, MLflow & Governance | 1:00 min |
| **Slide 3** | Data Hygiene & Target Leakage Prevention | Discovery of `CustomerFeedback` Leakage & Clean Ingestion | 1:00 min |
| **Slide 4** | Exploratory Data Analysis & Statistical Evidence | Contract Dominance, Tenure Decay, Service Mix Insights | 1:00 min |
| **Slide 5** | Missing Data Strategy: TotalCharges Imputer | Ridge Regression Champion Model vs. Deterministic Baselines | 1:00 min |
| **Slide 6** | MLflow Benchmark & Champion Classifier | 7 Algorithms Benchmarked, LightGBM Selection (ROC-AUC 0.844) | 1:00 min |
| **Slide 7** | Explainability & Core Risk Drivers | SHAP / Feature Importances (Contract, Tenure, Tech Support) | 1:00 min |
| **Slide 8** | Unstructured Voice: Sentiment & Thematic Analysis | Thiago's NLP Track: Separating Actionable Reasons from Leakage | 1:00 min |
| **Slide 9** | Actionable Retention Framework | Strategy Matrix: High-Risk vs. Tenure Tiers | 1:00 min |
| **Slide 10** | Business ROI & Transition to Live Demo | Financial Impact, Unit Economics & Handoff to Madu & Carol | 1:00 min |
| **Live Demo**| **Databricks Live Dashboard (Madu & Carol)** | **Interactive Customer Scoring, Risk Drivers & Retention Playbook** | **5:00 min** |

---

## Slide 1: Executive Summary & Business Problem
- **Headline:** Transforming Telecom Churn from Reactive firefighting to Proactive Retention.
- **The Core Problem:**
  - Customer acquisition cost (CAC) in telecom is 5× to 7× higher than retention cost.
  - Churn is concentrated in high-value monthly contracts and new subscribers, causing substantial Annual Recurring Revenue (ARR) erosion.
- **Our Solution:**
  - An end-to-end Machine Learning pipeline on Databricks Unity Catalog that predicts churn risk with **0.844 ROC-AUC**, explains the root drivers per customer, and equips frontline agents with retention actions via an interactive Databricks Dashboard.
- **Key Deliverables:**
  - Automated Delta Feature Store, 2 registered MLflow champion models, validated batch inference scores, and an operational dashboard for business teams.

---

## Slide 2: Modern Databricks Lakehouse Architecture
- **Headline:** Scalable, Reproducible, and Governed from Raw CSVs to Business Decisions.
- **Pipeline Architecture:**
  - **Medallion Ingestion:** Raw CSVs (`history.csv`, `inference.csv`) → Cleaned Delta tables (`workspace.default.train_silver`).
  - **Feature Store & Target Store:** Parameterized SQL queries producing modular tables in Unity Catalog.
  - **Two-Track ML Modeling:**
    - *Track 1:* Missing metric imputation (`TotalCharges` Ridge regressor).
    - *Track 2:* Churn classification benchmark & champion selection (`LightGBM`).
  - **Experiment Governance:** 100% tracked in MLflow (hyperparameters, CV metrics, artifacts, and champion aliases).
  - **Serving Layer:** Batch scored inference Delta tables directly powering the Databricks Dashboard.

---

## Slide 3: Data Quality, Schema Hygiene & Target Leakage
- **Headline:** Preventing Costly Production Blindspots Through Rigorous Data Auditing.
- **The CustomerFeedback Leakage Discovery:**
  - In `inference.csv`, the column `CustomerFeedback` contained post-facto cancellation comments (e.g., *"decided to cancel"*, *"recently churned"*).
  - **Critical Engineering Decision:** Identified and isolated this column as **pure target leakage**—strictly excluded from model features to protect production validity.
- **Handling Data Imperfections:**
  - **Discrepant Rows:** Investigated row count differences between raw sources and catalog tables; confirmed zero customer overlap between train and inference sets.
  - **`tenure == 'unknown'`:** Converted via `TRY_CAST` and imputed using robust median transformers rather than risky reverse calculations.

---

## Slide 4: Exploratory Analysis & Statistical Evidence
- **Headline:** The Data Tells a Clear Story: Contract Type and Tenure Define Retention.
- **Key Statistical Findings (Bruno’s Statistical Validation Layer):**
  - **Contract Tier Dominance:** Month-to-month (M2M) contracts exhibit **~43–50% churn**, while 2-year contracts drop churn below **3%**.
  - **Tenure Vulnerability:** The highest probability of attrition occurs in months **0 to 6**. Once a customer reaches month 24, churn drops drastically.
  - **Payment Friction:** Electronic check users show 2.5× higher churn than credit card or automated bank transfer users.
  - **Demographics:** Gender, SeniorCitizen status, and Partners showed negligible statistical correlation with churn, directing our focus purely to service quality, billing, and contract mechanics.

---

## Slide 5: Track 1 — Solving Missing Data: The TotalCharges Imputer
- **Headline:** Replacing Arbitrary Heuristics with MLflow Champion Regression.
- **The Challenge:**
  - `TotalCharges` had missing values corresponding to blank strings or brand-new clients (`tenure == 0`).
- **The Technical Evaluation:**
  - *Baseline Rule:* Bruno’s deterministic benchmark `tenure * MonthlyCharges` provided an intuitive ground truth baseline.
  - *Champion Model:* Raquel trained and tuned an **OLS vs. Ridge Regression Pipeline** within Databricks.
- **The Outcome:**
  - The Ridge Regressor was selected and registered in Unity Catalog as `workspace.default.total_charges_ridge_tuned@champion`.
  - Seamlessly re-applied in downstream ABT assembly to guarantee 100% complete data for inference scoring.

---

## Slide 6: Track 2 — Model Benchmarking & Champion Selection
- **Headline:** Rigorous 7-Algorithm Benchmark with Cross-Validation & Metric Alignment.
- **The Benchmark:**
  - Evaluated **7 algorithm families**: Logistic Regression, Random Forest, LightGBM, XGBoost, Multi-Layer Perceptron (MLP), Decision Trees, and KNN.
  - Evaluated across ROC-AUC, Average Precision (PR-AUC), Recall, F1-Score, and Brier Score.
- **Champion Results:**
  - **LightGBM** achieved the top score with **ROC-AUC: 0.844** and balanced generalization across folds.
  - Calibrated threshold optimization prioritizing **Recall** to minimize expensive False Negatives (unidentified churners).
  - Logged and promoted in MLflow as `@champion`.

---

## Slide 7: Model Explainability & Frontline Risk Drivers
- **Headline:** White-Box AI: Giving Attendants the Exact "Why" Behind Every Churn Score.
- **Top 5 Feature Importance Drivers (`df_importances.csv`):**
  1. **`nbrTenure` (186 score):** Length of relationship is the primary anchor of customer loyalty.
  2. **`ContractTier` (M2M vs. Long):** Contractual lock-in vs. monthly flexibility.
  3. **`MonthlyCharges` / Charge Band:** Price sensitivity relative to speed and product bundle.
  4. **`TechSupport` & `OnlineSecurity`:** Absence of support services sharply increases churn probability.
  5. **`isPaymentMethodCheck`:** Friction in manual check payments correlates with billing frustration.
- **Business Translation:** The model does not just output a number; it identifies whether the risk is financial, technical, or contractual.

---

## Slide 8: Unstructured Voice: Feedback & Sentiment Insights
- **Headline:** Thiago’s Track: Leveraging Unstructured Customer Feedback Beyond the Model.
- **Role in the Solution:**
  - While excluded from model features due to leakage, `CustomerFeedback` is a goldmine for **post-scoring diagnostics and qualitative root-cause analysis**.
- **Thematic Clusters Identified:**
  - **Price / Rate Hikes:** Frustration with bill spikes after promotion expiration.
  - **Competitor Fiber Offerings:** Customers leaving for faster upload speeds or bundled streaming.
  - **Customer Service Latency:** Repeat unresolved technical support tickets.
- **Operational Value:** Provides contextual coaching notes for call center supervisors alongside the quantitative churn score.

---

## Slide 9: Actionable Retention Framework & Business Playbook
- **Headline:** Prescriptive Analytics: Turning Probability Scores into Targeted Interventions.
- **Retention Strategy Matrix:**
  - **Tier 1: High Risk (>75%), Low Tenure (<6 mo):**
    - *Profile:* New month-to-month customers experiencing initial onboarding friction.
    - *Action:* Proactive onboarding call + free 3-month Tech Support add-on + incentive to switch to automated payment.
  - **Tier 2: High Risk (>75%), High Tenure (>24 mo):**
    - *Profile:* Long-term loyal customers reacting to sudden bill increases or competitor fiber offers.
    - *Action:* Exclusive loyalty renewal offer: 1-year contract discount locking in current rate.
  - **Tier 3: Moderate Risk (40–75%):**
    - *Profile:* Fiber optic users without security add-ons paying manual checks.
    - *Action:* Digital payment setup bonus + bundled Online Security trial.
  - **Rule of Thumb:** *Never offer costly discounts to unrecoverable or low-risk cohorts.*

---

## Slide 10: Financial Impact, Business ROI & Demo Hand-off
- **Headline:** Demonstrable Bottom-Line Value and Seamless Operationalization.
- **Unit Economics & Expected ROI:**
  - Identified ~370 high-probability churners in the unlabelled inference cohort.
  - Assuming an average ARPU of \$65/month (\$780/year) and a conservative 25% retention success rate:
    - **Estimated Annual Revenue Protected:** **~$72,000+** on this single batch alone.
- **Seamless Operational Handoff:**
  - To make this actionable every day for retention agents, **Maria Eduarda (Madu) and Caroline (Carol)** developed a fully interactive Databricks Dashboard.
  - **Transition:** *"I will now hand the floor to Madu and Carol for a 5-minute live demonstration of the Databricks Retention Dashboard!"*

---

## 5-Minute Live Dashboard Demo Script (Madu & Carol)

### **Minute 1: Dashboard Overview & Executive KPI Summary**
- **Presenter:** Madu
- **Screen:** Databricks Dashboard — Executive Overview Tab.
- **Script:**
  > *"Good afternoon everyone. Here is our live Databricks Retention Dashboard, connected directly to our Delta Lake inference scoring tables. At a glance, management sees our portfolio health: total active customers evaluated (1,409), predicted churn rate (26.3%), and the total monthly revenue at risk (\$24,000/mo). The top charts break down churn distribution across contract tiers and payment methods, confirming our model findings."*

### **Minute 2: Cohort Segmentation & Filter Capabilities**
- **Presenter:** Carol
- **Screen:** Cohort Filtering & Interactive Widgets.
- **Script:**
  > *"Retention teams don't work with raw CSVs—they need prioritized work queues. With the interactive controls on the left, an operations manager can filter customers by Risk Tier (e.g. 'High Risk >75%'), Contract Type ('Month-to-Month'), and Tenure (<12 months). The table immediately updates, ranking customers by expected revenue loss, enabling retention squads to tackle the highest-value churners first."*

### **Minute 3: Customer 360 Deep-Dive & Root-Cause Drivers**
- **Presenter:** Madu
- **Screen:** Customer Lookup Detail View (e.g., selecting customer `4226-KKDON`).
- **Script:**
  > *"When a customer service agent receives an inbound call or prepares an outbound retention campaign, they select the specific Customer ID. Instantly, the dashboard renders the customer's 360 profile: tenure (2 months), monthly charges (\$85.50), and predicted churn probability (82%). Most importantly, right below the score, the dashboard shows the Top 3 Risk Drivers computed by our LightGBM model: M2M Contract, Lack of Tech Support, and Electronic Check payment."*

### **Minute 4: Prescriptive Retention Offer Recommendation**
- **Presenter:** Carol
- **Screen:** Dynamic Playbook / Action Card.
- **Script:**
  > *"Instead of leaving retention offers to guesswork, our dashboard incorporates the business logic we defined in our project plan. Based on this customer's profile (New Customer + High Tech Frustration), the system recommends: 'Offer 1: Free 6-month Tech Support upgrade upon switching to Annual Contract' and 'Offer 2: \$5/month discount with Automatic Billing enrollment'. This maximizes retention ROI while guarding margin."*

### **Minute 5: Summary, Monitoring & Wrap-Up**
- **Presenter:** Madu & Carol
- **Screen:** Model Drift & Feedback Loop Tab.
- **Script:**
  > *"Finally, the dashboard tracks campaign conversion and feedback, closing the loop with our Unity Catalog feature store for ongoing model retraining. This completes our end-to-end journey from raw data to machine learning, and finally to business value on Databricks. Thank you, and we now open for your questions!"*
