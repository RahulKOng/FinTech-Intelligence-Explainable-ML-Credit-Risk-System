# FinTech: Explainable Machine Learning System for Loan Default and Credit Risk Assessment

An end-to-end, interpretable credit risk scoring and loan default analytics platform designed for consumer and commercial lending institutions. The system integrates machine learning discrimination power with strict regulatory explainability standards (Equal Credit Opportunity Act - **ECOA** / Fair Credit Reporting Act - **FCRA**).

---

## System Architecture: Three Core Sections

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           1. INGESTION LAYER                                 │
│  • CSV / JSON File Drag-and-Drop Uploader (Client & Server side)             │
│  • Kaggle Dataset Link / URL Parser (Auto-detects slug & schemas)            │
│  • Open Benchmark Datasets (LendingClub, German Credit, Give Me Some Credit) │
│  • Schema Inspector, Missing Value Profiler & Target Column Selector        │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           2. PROCESSING LAYER                                │
│  • Preprocessing & Imputation (Median imputation, One-Hot Encoding)          │
│  • Interpretable Logistic Scorecard (L2-regularized with odds ratios)        │
│  • Gradient Boosted Decision Tree (GBDT) Ensemble (Non-linear interactions)   │
│  • Calibrated FICO Credit Scoring (300 to 850 scale mapping, Basel II tiers) │
│  • Explainability Engine (Exact Shapley attributions & TreeSHAP)             │
│  • One-Click "⚡ Calculate & Evaluate" with live execution telemetry          │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   3. EXECUTIVE & EXPLAINABILITY DASHBOARD                    │
│  • Portfolio KPIs (AUC-ROC, Approval Rate, Expected Loss, Net Profit, ROI)   │
│  • Global SHAP Feature Importance Chart (Directional Risk vs Protective)     │
│  • Multi-Model ROC Curve Diagnostic Comparison                               │
│  • Calibrated FICO Score Distribution (Prime, Standard, Subprime, High Risk) │
│  • Dynamic Confusion Matrix & Risk Tolerance Cutoff Slider                   │
│  • Portfolio Financial Optimization Curve (Net Margin vs Defaults)           │
│  • Individual Loan Waterfall Explainer with ECOA Adverse Action Notices      │
│  • Counterfactual "What-If" Simulator for Instant Borrower Re-scoring       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical & Explainability Formulations

### 1. Basel II / FICO Credit Score Calibration
Loan default probability $P(\text{Default} \mid x) = p$ is converted into a standard FICO-scale credit score ($300 \le \text{Score} \le 850$):
$$\text{Odds} = \frac{1 - p}{p}$$
$$\text{Factor} = \frac{\text{PDO}}{\ln(2)}, \quad \text{Offset} = \text{BaseScore} - \text{Factor} \cdot \ln(\text{BaseOdds})$$
$$\text{Credit Score} = \text{Offset} + \text{Factor} \cdot \ln(\text{Odds})$$

**Calibration Parameters:**
- $\text{BaseOdds} = 19.0$ (corresponds to a 5.0% baseline default probability).
- $\text{BaseScore} = 720$ (Prime Tier).
- $\text{PDO} = 35$ (Points to Double the Odds: doubling the odds from 19:1 to 38:1 adds +35 points $\rightarrow$ 755 Super Prime).

### 2. Exact Shapley Value Attribution (Linear / Scorecard)
For linear scorecards in log-odds space $f(x) = \sum_{j=1}^M w_j x_j + b$, the exact Shapley attribution for feature $j$ satisfies efficiency:
$$\phi_j(x) = w_j \cdot (x_j - E[x_j])$$
$$\sum_{j=1}^M \phi_j(x) = f(x) - E[f(x)]$$

### 3. ECOA & FCRA Regulatory Adverse Action Generation
When a loan application is rejected or subjected to adverse pricing ($p > \text{Threshold}$), lenders are legally required by ECOA Regulation B and FCRA to provide the primary principal reasons:
1. Features are ranked by positive contribution to default log-odds ($\phi_j > 0$).
2. The top 4 factors are mapped to legally standardized reason codes (e.g. *Proportion of debt to monthly gross income is too high*, *Excessive revolving line utilization*, *Credit bureau risk score below lending threshold*).

---

## Project Structure

```
├── data/
│   ├── lending_club_sample.csv       # 1,200 Consumer loans (FICO, DTI, Revolving Util, etc.)
│   ├── german_credit_sample.csv      # 1,000 German Credit Statlog benchmark records
│   └── give_me_some_credit_sample.csv# 1,000 Kaggle credit benchmark records
├── src/
│   ├── preprocessing.py              # Schema detection, imputation, scaling, one-hot encoding
│   ├── models.py                     # Logistic Scorecard (L-BFGS-B) & Gradient Boosted Trees
│   ├── explainability.py             # Shapley attributions, Waterfall plots, Adverse Action codes
│   ├── scoring.py                    # FICO points mapping, Risk Grading (A-HR), Portfolio optimization
│   └── pipeline.py                   # End-to-end execution pipeline
├── web/
│   ├── index.html                    # Complete single-page dashboard with SVG/Canvas charts
│   └── server.py                     # Python HTTP server with /api/evaluate endpoint
├── tests/
│   └── test_engine.py                # Comprehensive automated unit & integration tests
├── artifacts/
│   ├── results.json                  # Complete pipeline output metrics & evaluations
│   └── plots/
│       ├── roc_curve.png             # Receiver Operating Characteristic plot
│       ├── shap_global_importance.png# Global SHAP feature importance plot
│       ├── confusion_matrix.png      # Confusion matrix heatmap
│       ├── cutoff_optimization.png   # Net profit vs cutoff threshold curve
│       └── applicant_waterfall.png   # Individual applicant risk waterfall decomposition
└── main.py                           # CLI pipeline execution script
```

---

## How to Run

### 1. Run the Python CLI & Generate Visualizations
```bash
python3 main.py --dataset data/lending_club_sample.csv --threshold 0.20
```
This trains the models, computes SHAP values, outputs executive metrics to terminal, and writes 5 PNG visualization plots into `artifacts/plots/`.

### 2. Run the Interactive Web Dashboard
Option A: Start the built-in server:
```bash
python3 web/server.py 8080
```
Open `http://localhost:8080` in your web browser.

Option B: Open `web/index.html` directly in any modern browser. The application is completely self-contained with embedded JavaScript calculation engines and SVG/Canvas visualizations, requiring zero external CDN connections.

### 3. Run Automated Tests
```bash
python3 tests/test_engine.py
```
All unit tests for preprocessing, ML modeling, explainability attribution, and credit scoring will execute.
