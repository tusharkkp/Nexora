# AI-Based Food Inflation Prediction & Household Affordability Monitoring System

## Complete Presentation Script, Architectural Walkthrough, Algorithm Encyclopedia & Viva Mastery Guide

> **Examination:** Third-Year B.Tech AIML Practical Examination (**30 Marks Total** — Components A–I, CO1–CO4, Bloom's Levels L3–L5)  
> **Problem Statement:** AI-Based Food Inflation Prediction and Household Affordability Monitoring System ([`Problem_statement.txt`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/Problem_statement.txt))  
> **Commodities Modeled:** Tomato, Onion, Moong Dal (Pulses), Sugar — with zero-shot fallback generalization for Potato, Rice, Wheat, or custom items  
> **Syllabus Coverage:** 100% of [`ML_algos.txt`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/ML_algos.txt) (22 checklist items implemented across 29+ trained models/experiments)  
> **Live Applications:** Streamlit Dashboard ([`app.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/app.py)), Gradio Web App ([`gradio_app.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/gradio_app.py)), CLI Inference Tools ([`scripts/test_unified_model.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/test_unified_model.py), [`scripts/test_model.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/test_model.py))  

---

## Table of Contents

1. [Spoken Presentation Script (Verbatim Flow for Your Live Demo)](#1-spoken-presentation-script-verbatim-flow-for-your-live-demo)
2. [Problem Statement, Intuition & Rubric Alignment (CO1–CO4)](#2-problem-statement-intuition--rubric-alignment-co1co4)
3. [Data Acquisition Architecture & Multi-Source Corpus (Component A — 2 Marks)](#3-data-acquisition-architecture--multi-source-corpus-component-a--2-marks)
4. [Descriptive Statistical Analysis & Patterns (Component B — 2 Marks)](#4-descriptive-statistical-analysis--patterns-component-b--2-marks)
5. [Exploratory Data Analysis & Visualizations (Component C — 2 Marks)](#5-exploratory-data-analysis--visualizations-component-c--2-marks)
6. [Data Preprocessing, Feature Engineering & Zero-Leakage Design (Component D — 2 Marks)](#6-data-preprocessing-feature-engineering--zero-leakage-design-component-d--2-marks)
7. [Chronological Splitting & Time-Series Cross-Validation (Component E — 2 Marks)](#7-chronological-splitting--time-series-cross-validation-component-e--2-marks)
8. [End-to-End Model Training, Dual-Model Paradigm & Tuning (Component F — 4 Marks)](#8-end-to-end-model-training-dual-model-paradigm--tuning-component-f--4-marks)
9. [Comprehensive Model Evaluation & Benchmarks (Component G — 8 Marks)](#9-comprehensive-model-evaluation--benchmarks-component-g--8-marks)
10. [Production Deployment, Engel's Law Engine & Ethics (Component H — 6 Marks)](#10-production-deployment-engels-law-engine--ethics-component-h--6-marks)
11. [Every ML Algorithm Explained (Short & Deep Reference — 22/22 Syllabus Items + Variants)](#11-every-ml-algorithm-explained-short--deep-reference--2222-syllabus-items--variants)
12. [Complete Glossary of Domain, Economic, Statistical & ML Terms](#12-complete-glossary-of-domain-economic-statistical--ml-terms)
13. [File-by-File Codebase Map & Notebook Walkthrough Guide](#13-file-by-file-codebase-map--notebook-walkthrough-guide)
14. [Hard Faculty Viva Questions & Model Answers (With Honest Tradeoffs)](#14-hard-faculty-viva-questions--model-answers-with-honest-tradeoffs)

---

## 1. Spoken Presentation Script (Verbatim Flow for Your Live Demo)

> **Tip:** Follow this 8–10 minute speaking flow while switching tabs between your Jupyter Notebooks (`notebooks/01`–`05`) and your running Streamlit app (`streamlit run app.py`).

### Opening & Hook (0:00 – 1:00)

> *"Good morning, respected faculty. Today I am presenting my Third-Year B.Tech AIML project titled **'AI-Based Food Inflation Prediction and Household Affordability Monitoring System'**.*
>
> *Most academic ML projects use standard static benchmark datasets like Iris or House Prices. Instead, I built a domain-specific **end-to-end econometric AI platform** from scratch using **189,413 real-world records** collected from four official Government of India portals—AGMARKNET, data.gov.in, the Department of Consumer Affairs (DCA), and MoSPI / NSSO.*
>
> *Why does this problem matter? In India, NSSO HCES 2022–23 data proves that the poorest 20% of households spend **56.6%** of their entire budget just on food, compared to **29.8%** for the richest 20%. When erratic monsoons or storage shortages cause tomato or onion prices to triple in a few weeks, low-income families immediately suffer nutritional distress. Our system bridges **agricultural supply-side AI forecasting** with **demand-side household economic vulnerability monitoring**."*

### Part 1 — Data Engineering & Why Real Data Is Hard (1:00 – 2:30)
*(Show `notebooks/01_data_collection_overview.ipynb`)*

> *"Let's look at **Notebook 01** covering Rubric Component A. Using our modular pipeline in `scripts/main.py`—via REST APIs (`mospi_client.py`, `data_gov_client.py`), HTML table scrapers (`relevant_scrapers.py`), and `pdfplumber` PDF table extraction—we built five standardized datasets totaling **189,413 records**:*
> 1. **`wholesale_prices.csv` (71,307 rows):** Daily APMC auction min, max, and modal prices in **₹/Quintal** across 1,108 mandis.*
> 2. **`market_arrivals.csv` (66,745 rows):** Physical trucked supply in **Metric Tonnes** totaling over 20.4 million tonnes—crucial because price is driven by supply.*
> 3. **`retail_prices.csv` (39,187 rows):** Consumer retail prices in **₹/Kg or ₹/Litre** across 60 urban centres.*
> 4. **`cpi_data.csv` (12,022 rows):** Official MoSPI CPI indices from 2014–2025.*
> 5. **`household_expenditure.csv` (152 rows):** NSSO HCES 2022–23 expenditure fractiles.*
>
> *Two crucial data-integrity rules guided our engineering:*
> - ***First, strict Retail vs. Wholesale isolation:** You cannot mix ₹/Quintal wholesale auction bids with ₹/Kg consumer prices because retail adds transport freight, mandi cess, spoilage, and 40–80% margins.*
> - ***Second, Zero Fabrication Policy:** When mandis close on Sundays or holidays, we never synthetically interpolate prices; we index strictly by active trading sessions."*

### Part 2 — Statistical Summary & EDA Insights (2:30 – 4:00)
*(Show `notebooks/02_exploratory_data_analysis.ipynb`)*

> *"In **Notebook 02** (Rubric Components B & C), we tested four sentinel commodities chosen deliberately across the perishability spectrum: **Tomato** (3–5 day shelf life), **Onion** (2–4 month storage), **Moong Dal** (protein staple), and **Sugar** (government-regulated sweetener).*
>
> *Computing seven descriptive statistics revealed why one-size-fits-all intuition fails:*
> - ***Tomato** has a Coefficient of Variation of **75.36%** on daily Pune sessions (mean ₹1,299.71 vs median ₹950.00/Qtl) and **+1.935 positive skewness**—prices stay low most of the year, then spike violently during the July–August monsoon washout.*
> - ***Onion** surges in October–November (**57.68% CV**) when stored rabi onions rot before the kharif harvest.*
> - ***Moong Dal** (**13.19% CV**) and **Sugar** (**10.75% CV**) trade in tight, predictable bands.*
>
> *Crucially, in agricultural economics, those upper outliers in our Tukey boxplots are **not errors to drop**—they are the exact inflation crises we want to predict! Meanwhile, our correlation heatmap confirmed both strong short-term autoregression and severe multicollinearity (`r > 0.95` among price lags and rolling means), directly motivating **regularized models like Ridge and Lasso** over plain OLS."*

### Part 3 — Feature Engineering & Leakage Prevention (4:00 – 5:15)
*(Show `notebooks/03_data_preprocessing.ipynb`)*

> *"The biggest trap in time-series ML is **lookahead data leakage**—for instance, using `train_test_split(shuffle=True)` or unshifted rolling windows. If January test rows sit beside February training rows that share a 7-day rolling mean, you get a fake 99% score that collapses in production.*
>
> *In **Notebook 03** and `scripts/build_features.py` (Components D & E), we engineered **16 strictly backward-looking numeric features** across three groups: autoregressive lags, shifted 7-day rolling statistics (`shift(1).rolling(7)` so today's target never peeks forward), supply arrival quantities, and monsoon/calendar indicators—plus `commodity_name`, `category`, and `is_perishable` when pooling into our **4,480-row Unified Feature Store**. We then partitioned chronologically—**70% Train (3,133 rows), 15% Validation (672 rows), and 15% Test (675 rows)**—and used expanding-window **`TimeSeriesSplit`** for grid search."*

### Part 4 — Algorithms, Training & Key Discoveries (5:15 – 7:15)
*(Show `notebooks/04_regression_model_development.ipynb` and `notebooks/05_model_evaluation.ipynb`)*

> *"Across **Notebooks 04 and 05** (Components F & G), we implemented every single algorithm category listed in `ML_algos.txt`—spanning 22 checklist items across regression, classification, ensembles, optimization, clustering, and PCA—and adopted a **Dual-Model Strategy**:*
> 1. ***Universal Multi-Commodity Pipeline (`Ridge L2`, $\alpha=1.0$):** Combines `OneHotEncoder(handle_unknown='ignore')` and `StandardScaler` inside a `ColumnTransformer`. On 675 completely unseen test sessions spanning all four foods, it achieved **$R^2 = 0.9935$**, **MAE = ₹65.97/Qtl (~₹0.66/Kg)**, and **MAPE = 6.65%**—predicting Moong Dal within **0.36% MAPE** and Tomato/Onion within ~**10% MAPE**.*
> 2. ***Dedicated Crop Specialists:** For **Tomato**, **Lasso ($L_1$)** won with **RMSE ₹93.35/Qtl** and **$R^2 = 0.8454$** (beating Naive Persistence ₹112.11) by zeroing out redundant correlated features (`price_lag_3`, `rolling_arrival_mean_7`, `is_monsoon_season`). For **Onion**, **SVR with an RBF kernel** won with **RMSE ₹179.06/Qtl** (beating Baseline ₹191.32) by modeling non-linear storage supply shocks.*
>
> *Notice a fascinating empirical finding: **Linear regularized models outperformed Random Forest and Gradient Boosting** here. Why? Because tree ensembles cannot extrapolate outside the training price envelope, whereas Ridge and Lasso naturally follow linear autoregressive trends.*
>
> *We also benchmarked Gradient Descent optimizers (**Mini-batch GD batch=32** achieved the best test RMSE of **₹93.58/Qtl** over Batch GD ₹94.23 and SGD ₹95.23), 9 early-warning **Price Spike Classifiers** (where Gaussian Naive Bayes achieved **96.49% Recall** as a high-sensitivity safety net and Stacking reached **63.16% Accuracy** on Onion), and **Unsupervised Regime Segmentation** (K-Means $k=2$, Ward Hierarchical, DBSCAN isolating **71 anomalous trading days**, and PCA reducing 15 features to **8 components explaining 97.81% of variance**)."*

### Part 5 — Live Production App & Engel's Law Demo (7:15 – 9:00)
*(Switch to live Streamlit App `streamlit run app.py`)*

> *"Finally, for **Rubric Component H (Deployment Readiness)**, I wrapped our serialized `joblib` pipelines and econometric calculator into a production-ready **Streamlit dashboard (`app.py`)** plus a free **Gradio version (`gradio_app.py`)** ready for Hugging Face Spaces.*
>
> *Let's test Module 1 live: if I select **Tomato** at ₹14/kg with 150 tonnes arriving in July, our Universal Ridge pipeline outputs tomorrow's wholesale and retail forecast along with a Surge/Crash/Stable alert in under 5 milliseconds without retraining.*
>
> *Now switching to Module 2—**Household Food Affordability & Engel's Law Monitor**: if a household adjusts income—say from ₹30,000 down to ₹12,000—or increases grocery prices, our deterministic engine (`src/affordability.py`) computes the Food Budget Share percentage, Discretionary Balance Left, and plots the family's position directly against **NSSO HCES 2022–23 Rural (46.38%), Urban (39.17%), and Bottom-20% (56.6%)** vulnerability thresholds.*
>
> *Everything is backed by automated `pytest` unit tests (`tests/`), strict input validation, zero-PII privacy, and clear ethical disclaimers. Thank you—I am ready for your questions."*

---

## 2. Problem Statement, Intuition & Rubric Alignment (CO1–CO4)

### Core Intuition Behind Building This System

Most price-prediction demos stop at predicting a number ($R^2 = 0.85$) and ignore what that price jump actually does to families. Conversely, economic surveys report inflation months after the damage is done. Our system connects both sides:

```
  [Agricultural Supply & Prices]  ──(ML Forecasting)──▶  [Tomorrow's Price & Spike Alert]
                                                                    │
                                                        (Wholesale → Retail Impact)
                                                                    ▼
  [Family Income & Grocery Basket] ─(Engel's Law Engine)─▶ [Household Food Vulnerability Alert]
```

### Complete 30-Mark Rubric Mapping ([`rubrics.md`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/rubrics.md))

| Component | Marks | CO / Bloom | What We Built to Satisfy It | Primary Artifacts |
|:---|:---:|:---:|:---|:---|
| **A. Data Collection** | 2 | CO1 / L3 | Multi-source government corpus (**189,413 records** across 5 datasets), REST APIs + HTML scraping + PDF table extraction | [`scripts/main.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/main.py), [`01_data_collection_overview.ipynb`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/notebooks/01_data_collection_overview.ipynb) |
| **B. Statistical Summary** | 2 | CO1 / L3 | **7+ statistical operations** (Mean, Median, Std Dev, Variance, Min/Max, Quartiles/IQR, CV, Skewness, Kurtosis) across 4 food commodities | [`02_exploratory_data_analysis.ipynb`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/notebooks/02_exploratory_data_analysis.ipynb), [`reports/statistical_analysis.md`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/statistical_analysis.md) |
| **C. EDA & Visualization** | 2 | CO1 / L3 | **5 distinct visualization families** in EDA + **11 modeling diagnostic plots** (16 total PNGs at 300 DPI) | `reports/figures/*.png`, [`02_exploratory_data_analysis.ipynb`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/notebooks/02_exploratory_data_analysis.ipynb) |
| **D. Data Preparation** | 2 | CO1 / L3 | Unit isolation (₹/Qtl vs ₹/Kg), anomalous bound flagging, deduplication, **16 zero-leakage backward features**, multi-commodity pooling (`4,480` rows) | [`scripts/build_features.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/build_features.py), [`03_data_preprocessing.ipynb`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/notebooks/03_data_preprocessing.ipynb) |
| **E. Splitting & CV** | 2 | CO2 / L3 | Strict chronological **70/15/15 Train/Val/Test** partitioning (`3,133 / 672 / 675`), scalar fitting on train-only, expanding `TimeSeriesSplit` | [`03_data_preprocessing.ipynb`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/notebooks/03_data_preprocessing.ipynb), [`scripts/train_and_evaluate.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/train_and_evaluate.py) |
| **F. ML Implementation** | 4 | CO2 / L4 | All 22 syllabus items in `ML_algos.txt` across 29+ trained models/experiments (Regression, Classification, Ensembles, GD, Clustering, PCA) + hyperparameter tuning | [`scripts/train_all_models.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/train_all_models.py), [`scripts/train_unified_model.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/train_unified_model.py), [`04_regression_model_development.ipynb`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/notebooks/04_regression_model_development.ipynb) |
| **G. Model Evaluation** | 8 | CO3 / L5 | Unseen test evaluation (MAE, MSE, RMSE, $R^2$, MAPE, Accuracy, Precision, Recall, F1, ROC-AUC, Silhouette, Davies-Bouldin), baseline comparisons, residual diagnostics | [`05_model_evaluation.ipynb`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/notebooks/05_model_evaluation.ipynb), `reports/*.csv` |
| **H. Deployment & Ethics** | 6 | CO4 / L5 | Serialized `joblib` + JSON contracts, zero-retraining inference engine, Streamlit 4-module app, free Gradio HF Spaces app, Engel's Law analyzer, `pytest` tests, ethical disclosures | [`app.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/app.py), [`gradio_app.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/gradio_app.py), [`src/inference.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/src/inference.py), [`src/affordability.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/src/affordability.py) |
| **I. Presentation & Oral** | 4 | CO4 / L5 | Live presentation script, complete viva preparation, honest limitations defense | **This file (`PRESENTATION_SCRIPT.md`)**, [`reports/viva_questions_and_answers.md`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/viva_questions_and_answers.md) |

---

## 3. Data Acquisition Architecture & Multi-Source Corpus (Component A — 2 Marks)

### How Data Was Acquired (`scripts/main.py`)

We built a reproducible multi-channel collection framework (`py -3.12 scripts/main.py --all`):

1. **Official REST API Clients (`scripts/api/`):**
   - [`mospi_client.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/api/mospi_client.py): Queries `https://api.mospi.gov.in/api` (`cpi/getCPIIndex`, `cpi/getItemIndex`, `hces/getHcesRecords`) with session pooling, pagination, rate-limiting (`0.5s` delay), and exponential backoff retries.
   - [`data_gov_client.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/api/data_gov_client.py): Queries Open Government Data resource `9ef84268-d588-465a-a308-a864a43d0070` ("Current Daily Price of Various Commodities from Various Markets") using `DATA_GOV_IN_API_KEY` from `.env`.
   - [`ceda_client.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/api/ceda_client.py): Implements CEDA Ashoka University's Agmarknet OpenAPI 3.0 spec (`api.ceda.ashoka.edu.in/v1`).
2. **HTML Web Scrapers & Harvesters ([`scripts/scraping/relevant_scrapers.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/scraping/relevant_scrapers.py)):**
   - Parses daily & historical retail bulletins from Department of Consumer Affairs (`fcainfoweb.nic.in`) using `BeautifulSoup`.
   - Ingests historical multi-year APMC mandi CSV archives (2016–2018+) for Maharashtra hubs (Pune, Lasalgaon/Nashik, Mumbai, Nagpur).
   - Respects ethical boundaries: **never automates CAPTCHA solving** on gated interactive reports or RBI's F5 anti-bot challenge.
3. **PDF Tabular Extraction:**
   - Automatically downloads `Factsheet_HCES_2022-23.pdf` from MoSPI and parses Statements **1, 4, 5, and 8** programmatically using **`pdfplumber`**.
4. **Standardization & Quality Audit (`scripts/processing/`):**
   - [`normalize_data.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/processing/normalize_data.py): Maps noisy commodity spellings (e.g., `"Green Gram Dal(Moong Dal)"` $\to$ `"Green Gram Dal (Moong Dal)"` / `"Moong Dal"`), parses mixed date formats (`DD/MM/YYYY`, `YYYY-MM-DD`, `Month YYYY`) to ISO 8601 `YYYY-MM-DD`, and cleans numeric strings (`"--"`, `"NR"` $\to$ `None`).
   - [`clean_data.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/processing/clean_data.py): Deduplicates, maps markets to districts (e.g., `Pune(Pimpri)` $\to$ `Pune`, `Lasalgaon` $\to$ `Nashik`), flags anomalous bids (`min > max` or `modal` outside `[min, max]`) via `is_anomalous_flag`, and writes the 5 processed CSVs.
   - [`validate_data.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/processing/validate_data.py): Produces [`reports/data_quality_report.md`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/data_quality_report.md).

### Verified Inventory of All 5 Standardized Datasets (**189,413 Total Records**)

| # | Processed File | Source Portal | Records | Date Range | Unit & Granularity | Key Role in System |
|:---:|---|---|---:|---|---|---|
| 1 | `wholesale_prices.csv` | AGMARKNET & data.gov.in API | **71,307** | 2016-01-01 to 2026-11-17 | **₹/Quintal** (100 kg), 1,108 mandis across 326 districts (93.9% Maharashtra) | **Primary ML modeling target & lag features** (`modal_price_rs_per_qtl`) |
| 2 | `market_arrivals.csv` | AGMARKNET Mandi Arrivals | **66,745** | 2016-01-01 to 2018-12-31 | **Metric Tonnes**, 109 MH mandis, **20,427,210 T** logged | **Exogenous supply curve ($Q_t$)** — merged with wholesale prices |
| 3 | `retail_prices.csv` | Dept. of Consumer Affairs (DCA) | **39,187** | 2010-01-01 to 2026-11-17 | **₹/Kg** or **₹/Litre**, 41 commodities, 60 urban centres | Tracks consumer-facing markup ("rockets & feathers" effect), strictly isolated from wholesale |
| 4 | `cpi_data.csv` | MoSPI CPI API | **12,022** | 2014-01-01 to 2025-12-01 | **CPI Index (Base 2012=100)** & YoY %; Rural/Urban/Combined (MH & All-India) | Macroeconomic headline food inflation context |
| 5 | `household_expenditure.csv` | NSSO HCES 2022–23 Factsheet | **152** | NSS 79th Round (2022–23) | **MPCE (₹)** & Budget Share (%), 12 fractiles & 36+ States/UTs | Calibrates **Engel's Law** affordability thresholds (`39.17%` Urban, `46.38%` Rural, `56.60%` Bottom-20%) |

### Why We Selected These Four Sentinel Commodities

| Commodity | Type & Shelf Life | Why Selected (Intuition) |
|---|---|---|
| **Tomato** | Perishable Vegetable (3–5 days) | Extreme monsoon washout volatility; **hardest** short-horizon stress test (**CV = 75.36%**). |
| **Onion** | Semi-Perishable Bulb (2–4 months) | Storage-loss volatility (rabi *chal* rotting before kharif arrival); high political/policy sensitivity (**CV = 57.68%**). |
| **Moong Dal** | Non-Perishable Pulse (12+ months) | High-value protein staple (~₹7,041/Qtl baseline) — tests high-price-scale generalization (**CV = 13.19%**). |
| **Sugar** | Government-Regulated Sweetener | Governed by cane FRP & mill release quotas (**CV = 10.75%**) — tests behavior on tightly bounded, near-flat series. |

---

## 4. Descriptive Statistical Analysis & Patterns (Component B — 2 Marks)

In **Notebook 02** (`02_exploratory_data_analysis.ipynb`, Code Cell 5), we executed **8+ statistical operations** on daily modal wholesale prices (Pune APMC for Tomato/Onion; Maharashtra daily median for Moong Dal/Sugar):

| Commodity | Daily Sessions ($N$) | Mean (₹/Qtl) | Median $Q_2$ (₹/Qtl) | Std Dev $\sigma$ (₹/Qtl) | Min / Max (₹/Qtl) | IQR ($Q_3-Q_1$) | CV ($\frac{\sigma}{\mu}\times 100$) | Skewness ($\gamma_1$) |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| **Tomato** | 1,047 | **1,299.71** | **950.00** | **979.50** | 400 / 5,765 | 750.00 | **75.36%** | **+1.935** |
| **Onion** | 1,047 | **1,046.32** | **801.00** | **603.49** | 151 / 3,663 | 490.00 | **57.68%** | **+1.748** |
| **Moong Dal** | 863 | **7,041.28** | **6,700.00** | **928.50** | 5,800 / 9,525 | 900.00 | **13.19%** | **+1.358** |
| **Sugar** | 830 | **3,552.76** | **3,667.00** | **381.88** | 2,725 / 6,500 | 571.25 | **10.75%** | **+0.759** |

*(Additional reference from **Notebook 01** across all raw mandi rows in `wholesale_prices.csv`: Tomato 29,691 rows mean ₹1,202.59 CV 86.69%; Onion 34,414 rows mean ₹1,005.57 CV 68.97%; Moong Dal 1,204 rows mean ₹7,118.81 CV 14.47%; Sugar 830 rows CV 10.75%. Retail benchmarks in `retail_prices.csv`: Tomato ₹19.02/kg CV 56.10%, Onion ₹16.92/kg CV 48.29%, Moong Dal ₹68.27/kg CV 15.22%, Sugar ₹33.55/kg CV 12.21%.)*

### Four Core Statistical Inferences

1. **Perishability Controls Relative Volatility ($CV$):** Moong Dal has a larger raw standard deviation (₹928.50) than Onion (₹603.49), yet Moong Dal is nearly **4.4× less volatile in percentage terms** ($CV = 13.19\%$ vs $57.68\%$) because its mean price plateau is 7× higher. **Lesson:** Never compare raw standard deviations across differently priced goods—always normalize via **Coefficient of Variation ($CV$)**.
2. **Mean $\gg$ Median Proves Right-Skewness:** For Tomato, Mean (₹1,299.71) is **36.8% higher** than Median (₹950.00) with **$\gamma_1 = +1.935$**. Most of the year tomatoes trade cheaply, but rare supply crunches stretch the upper tail above ₹5,700/Qtl.
3. **Price-Scale Separation Demands Feature Encoding:** Moong Dal averages ~₹7,041/Qtl while Onion averages ~₹1,046/Qtl (a **6.7× offset**). A single universal model must learn separate baseline intercepts via **One-Hot Encoding** (`commodity_name`, `category`).
4. **Outliers Must Not Be Deleted:** High prices exceeding Tukey's upper fence ($Q_3 + 1.5 \times \text{IQR}$) are authentic monsoon/storage supply crises, not sensor glitches. Deleting them would blind the model during actual inflation surges.

---

## 5. Exploratory Data Analysis & Visualizations (Component C — 2 Marks)

### A. 5 EDA Visualization Methods (Notebook 02 & `scripts/generate_eda.py`)

1. **Historical Multi-Commodity Time-Series Plots** ([`01_tomato_price_timeseries.png`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/01_tomato_price_timeseries.png)): Plots daily modal prices + 30-day rolling mean on separate vertical axes (preventing ₹7,000 Moong Dal from compressing ₹1,000 vegetables). Shows violent mid-year Tomato surges (reaching ₹5,700/Qtl in July 2017) vs. steady Moong/Sugar regimes.
2. **Histograms with Kernel Density Estimation (KDE)** ([`02_tomato_price_distribution.png`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/02_tomato_price_distribution.png)): Demonstrates heavy right-tail non-normality in Tomato/Onion (Shapiro-Wilk $p < 0.001$) vs. tighter bell profile in Sugar.
3. **Tukey Boxplots & Outlier Fences** ([`03_tomato_price_boxplot.png`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/03_tomato_price_boxplot.png)): Visualizes $Q_1, Q_2, Q_3$ boxes, IQR, and upper outlier clusters (`1.5 × IQR` rule). Proves upper whiskers capture genuine inflation regimes while lower outliers are absent (prices cannot fall below zero).
4. **Monthly Crop Calendar Seasonality Bar Charts** ([`04_tomato_monthly_seasonality.png`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/04_tomato_monthly_seasonality.png)): Exposes India's dual agricultural vulnerability: **Tomato peaks July–August** (monsoon transit/vine damage, February is cheapest) while **Onion peaks October–November** (stored rabi stocks deplete before late-kharif arrivals).
5. **Pearson Correlation Heatmap** ([`05_tomato_correlation_heatmap.png`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/05_tomato_correlation_heatmap.png)): Quantifies:
   - Strong **autoregressive signal** (`price_lag_1` & rolling mean correlate at **$r \approx 0.89–0.94$** with target price).
   - **Negative supply-price elasticity** (`arrival_quantity_tonnes` correlates at **$r = -0.18$** with modal price—more supply lowers prices).
   - **Multicollinearity alert** (`price_lag_1`, `price_lag_3`, `rolling_mean_7` correlate at $r > 0.95$ with each other), giving mathematical justification for **Ridge ($L_2$) and Lasso ($L_1$) regularization**.

### B. 11 Downstream Modeling Diagnostic Plots (`reports/figures/06–15` + Unified)

- `06_actual_vs_predicted_test.png`: Overlaid actual vs. top-5 regression trajectories on unseen dates.
- `07_residual_distribution.png` & `08_residuals_vs_predicted.png`: Gauss-Markov normality (mean error near zero: +₹1.02/Qtl on unified test) & homoscedasticity check.
- `09_model_metrics_comparison.png`: Bar charts comparing MAE, RMSE, and $R^2$ across all regression algorithms.
- `10_gradient_descent_convergence.png`: Log-scale MSE descent over 200 epochs comparing smooth Batch GD vs. noisy SGD vs. fast Mini-batch GD.
- `11_roc_curves_classification.png` & `12_confusion_matrices.png`: ROC-AUC curves and 2×2 confusion heatmaps for price spike classifiers.
- `13_kmeans_elbow_silhouette.png`, `14_hierarchical_dendrogram.png`, `15_pca_scree_and_clusters.png`: Unsupervised validation (Elbow + Silhouette $k=2$, Ward linkage dendrogram, 2D PCA cluster scatter + cumulative variance scree curve).
- `unified_multi_commodity_evaluation.png`: 4-quadrant 1:1 identity scatter across **Moong Dal, Onion, Sugar, and Tomato**.

---

## 6. Data Preprocessing, Feature Engineering & Zero-Leakage Design (Component D — 2 Marks)

### Step 1: Data Cleaning Rules & Why Each Was Chosen

| Cleaning Operation | Implementation (`clean_data.py` / NB03) | Why Selected |
|---|---|---|
| **Price-Type & Unit Segregation** | Separate CSVs & models for `wholesale_prices.csv` (₹/Qtl) vs `retail_prices.csv` (₹/Kg) | Mixing supply-chain stages or dividing wholesale by 100 without accounting for 40–80% retail markup/wastage corrupts predictions. |
| **Holiday Gap Preservation** | Index by active trading sessions (`sort_values('observation_date')`), **no Sunday zero/linear imputation** | APMCs officially close Sundays/holidays; inventing synthetic Sunday bids violates zero-fabrication integrity. |
| **Anomalous Bound Flagging** | Check `min > max` or `modal` outside `[min, max]` $\to$ `is_anomalous_flag` | Clerical transpositions are tracked explicitly rather than silently corrupting statistical bounds. |
| **Daily Multi-Yard Aggregation** | Group by `observation_date`: `min` of mins, `max` of maxes, `median` of modal prices, `sum` of arrivals | Eliminates duplicate same-day rows across sub-yards (e.g., Pune Gultekdi + Pimpri + Hadapsar = 168 duplicates removed in NB03). |
| **NaN Tail/Head Drop** | Drop initial rows lacking 7-session history + final row lacking $P_{t+1}$ target | Cannot train lag models without complete lookback context; zero synthetic forward-filling. |

### Step 2: All 18 Input Features (Unified Model) / 15 Features (Single-Crop Models)

> **Crucial Zero-Leakage Implementation Rule:** In `scripts/build_features.py` (which produced `model_ready_unified_commodities.csv`, Onion, and the 1,039-row Tomato corpus) and Notebook 03, rolling window features execute `.shift(1).rolling(window=7, min_periods=4)` so window aggregates cover $[t-7 \dots t-1]$ without touching future target $P_{t+1}$!

| # | Feature Name | Formula (`build_features.py`) | Category | Econometric Intuition |
|---:|---|---|---|---|
| 1 | `price_lag_1` | `P.shift(1)` *(previous session)* | Autoregressive | Captures immediate short-term price anchor |
| 2 | `price_lag_3` | `P.shift(3)` | Autoregressive | 3-session price anchor |
| 3 | `price_lag_7` | `P.shift(7)` | Autoregressive | Full 1-week trading cycle baseline |
| 4 | `rolling_mean_7` | `P.shift(1).rolling(7).mean()` | Rolling Stat | Smoothed 7-day trend free of daily bid noise |
| 5 | `rolling_std_7` | `P.shift(1).rolling(7).std()` | Rolling Stat | Local volatility regime (high $\sigma$ $\implies$ instability) |
| 6 | `rolling_min_7` | `P.shift(1).rolling(7).min()` | Rolling Stat | Recent 7-day support floor |
| 7 | `rolling_max_7` | `P.shift(1).rolling(7).max()` | Rolling Stat | Recent 7-day resistance ceiling |
| 8 | `price_momentum_7` | `P(t) - P.shift(7)` | Velocity / Trend | Net 7-session directional velocity (+ rising, − falling) |
| 9 | `daily_spread` | `max_price - min_price` | Intra-Day Dispersion | Quality heterogeneity / uncertain auction bidding range |
| 10 | `arrival_tonnes` | Sum of `arrival_quantity_tonnes` | Supply-Side ($Q_t$) | Immediate physical supply entering mandi (`0` if unavailable for Sugar/Moong) |
| 11 | `arrival_lag_1` | `arrival_tonnes.shift(1)` | Supply-Side | Yesterday's inflow overhang affecting trader inventory today |
| 12 | `rolling_arrival_mean_7` | `arrival_tonnes.shift(1).rolling(7).mean()` | Supply-Side | Weekly glut vs. scarcity baseline |
| 13 | `month` | Calendar month (`1–12`) | Calendar | Encodes annual crop harvest/monsoon months |
| 14 | `day_of_week` | `0` (Mon) to `6` (Sun) | Calendar | Post-weekend auction restocking dynamics |
| 15 | `is_monsoon_season` | `1` if `month in [6,7,8,9]`, else `0` | Calendar / Weather | Binary southwest monsoon shock flag |
| 16 | `is_perishable` *(Unified)* | `1` (Tomato/Onion), `0` (Sugar/Moong) | Structural Property | Teaches global model perishable vs. storable elasticity |
| 17 | `commodity_name` *(Unified)* | Categoricals: Tomato, Onion, Moong Dal, Sugar | Categorical ID | One-Hot encoded baseline price offset per commodity |
| 18 | `category` *(Unified)* | Vegetables, Pulses, Sugars & Sweets | Categorical Group | Hierarchical category intercept (`handle_unknown='ignore'` for unseen crops) |

*(Note on Target Variables: Regression target `target_price_t_plus_1 = P.shift(-1)`; Classification target `is_price_spike = 1` if `|(P_{t+1} - P_t) / P_t| × 100% > 10%`, else `0`, split into `'Surge'` / `'Crash'` / `'Normal'` in `spike_direction`. Secondary `target_price_t_plus_7 = P.shift(-7)` is also computed for comparison.)*

### Step 3: Pooling into Unified Multi-Commodity Corpus (`4,480` Rows)

In `scripts/build_features.py` (`build_unified_dataset()`), individually cleaned & chronologically split commodity series are pooled into [`model_ready_unified_commodities.csv`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/data/processed/model_ready_unified_commodities.csv) (**4,480 rows × 37 columns**):

- **Onion:** **1,765** rows (Pune APMC 1,007 + Lasalgaon APMC 758)
- **Tomato:** **1,039** rows (Pune APMC & sub-markets)
- **Moong Dal:** **854** rows (Maharashtra APMCs)
- **Sugar:** **822** rows (Maharashtra APMCs)
- **Note on Single-Yard Tomato Script:** [`scripts/preprocess_and_prepare_data.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/preprocess_and_prepare_data.py) builds an **873-row** exact-`"Pune"` subset (`model_ready_pune_tomato.csv`: `611 Train / 130 Val / 132 Test`) using `price_lag_1 = P(t)` as today's anchor, while `build_features.py` pools Pune sub-yards (Gultekdi/Pimpri/Hadapsar) yielding `1,039` rows (`156` test sessions, used to fit `price_prediction_model_tomato.joblib`). Similarly, NB03's inline demonstration groups Onion Pune+Lasalgaon onto one daily timeline producing `3,755` rows, while the production unified feature store (`4,480` rows) preserves Pune and Lasalgaon Onion as distinct series for higher volume.

---

## 7. Chronological Splitting & Time-Series Cross-Validation (Component E — 2 Marks)

### Why Random Splitting (`train_test_split(shuffle=True)`) Is Strictly Prohibited

Random splitting assumes observations are **Independent and Identically Distributed ($i.i.d.$)**. In commodity time series, adjacent sessions are autocorrelated ($r > 0.89$):

```
WRONG (Random Shuffling):
  Day 10 [Train] ──▶ Day 11 [Test] ──▶ Day 12 [Train]
  Problem: Day 12 overlaps with Day 11's rolling_mean_7 & price_lag_1 → LOOKAHEAD LEAKAGE!

CORRECT (Strict Chronological Partitioning):
  |◄────── 70% Train ──────►|◄── 15% Validation ──►|◄── 15% Unseen Test ──►|
        (Oldest Dates)           (Hyperparam Tuning)      (Newest Future Dates)
```

### Verified Partition Counts & Scaling Discipline

- **Unified Feature Store (`4,480` rows):** **Train = 3,133** (Moong 597, Onion 1,234, Sugar 575, Tomato 727) | **Validation = 672** (Moong 128, Onion 265, Sugar 123, Tomato 156) | **Test = 675** (Moong 129, Onion 266, Sugar 124, Tomato 156). Note that each commodity was split chronologically *before* pooling so no future crop leak occurs across commodities either!
- **Feature Scaling Discipline:** `StandardScaler().fit()` is called **strictly on Training features** ($\mu_{\text{train}}, \sigma_{\text{train}}$), and `.transform()` applies those training parameters to Validation and Test sets—preventing future distribution leakage. Inside [`ColumnTransformer`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/train_unified_model.py), this guarantees zero leakage even inside CV folds.
- **Cross-Validation (`TimeSeriesSplit(n_splits=5)`):** Standard $k$-fold shuffles past and future folds. We use **expanding-window `TimeSeriesSplit`** (shown in [`scripts/train_and_evaluate.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/train_and_evaluate.py)) where Fold 1 trains on early months to validate the next forward window, Fold 2 expands the training window forward, and no training fold ever peeks ahead of its validation block.

---

## 8. End-to-End Model Training, Dual-Model Paradigm & Tuning (Component F — 4 Marks)

### How Models Were Trained & Tuned

1. **Dual-Model Production Strategy:**
   - **Global Universal Model ([`scripts/train_unified_model.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/train_unified_model.py)):** Trains one `Pipeline([('preprocessor', ColumnTransformer), ('regressor', model)])` on `4,480` pooled rows across 8 candidate regressors (`MLR`, `Ridge`, `Lasso`, `ElasticNet`, `Decision Tree`, `Random Forest`, `Gradient Boosting`, `SVR`). `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` encodes `commodity_name` + `category` (7 columns) while `StandardScaler` normalizes the 16 numeric columns $\to$ **23 features fed into Ridge ($\alpha=1.0$)**.
   - **Dedicated Crop Specialists ([`scripts/train_all_models.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/train_all_models.py)):** Trains all **22 syllabus items from `ML_algos.txt`** across **29+ individual model runs** (1 Baseline + 12 Regressors + 3 GD Optimizers + 9 Classifiers + 3 Clustering + 1 PCA) on both **Tomato** (`price_prediction_model_tomato.joblib`) and **Onion** (`price_prediction_model_onion.joblib`), writing standardized artifacts to `models/` and `reports/*.csv`.
2. **Hyperparameter Tuning Choices:**
   - **Polynomial Regression (`train_and_evaluate.py`):** Tuned degree $d \in \{1, 2, 3\}$ and Ridge penalty $\alpha \in \{0.01, 0.1, 1.0, 10.0, 100.0\}$. Expanding 15 features to $d=2$ generates **135 terms** and $d=3$ generates **800+ terms**, causing overfitting on $\sim$600–880 rows (**Curse of Dimensionality**). We constrained polynomial expansion to **$d=2$ paired with Ridge $\alpha=10.0$**.
   - **Ridge ($\alpha=1.0$) & Lasso ($\alpha=1.0$ single / $\alpha=0.5$ unified, `max_iter=5000`):** Tuned to damp $r > 0.95$ lag multicollinearity. Inspecting `price_prediction_model_tomato.joblib` (`Lasso`) shows it automatically zeroed out **`price_lag_3 = 0.0`**, **`rolling_arrival_mean_7 = 0.0`**, and **`is_monsoon_season = 0.0`**, focusing weight on `price_lag_7 (+418.86)`, `price_momentum_7 (+269.08)`, `rolling_mean_7 (+243.76)`, and `price_lag_1 (+192.24)`, while `arrival_lag_1 (-28.43)` and `arrival_tonnes (-14.51)` carry **negative economic supply coefficients**!
   - **Tree Depth & Regularization:** Decision Trees capped at `max_depth=5` (`min_samples_split=10`) for single-crop and `max_depth=10` for unified; Random Forests capped at `n_estimators=100, max_depth=12`.
   - **Imbalanced Spike Classifiers:** Because $\pm 10\%$ jumps are minority events, `LogisticRegression`, `DecisionTreeClassifier`, and `SVC` use `class_weight='balanced'` so classifiers don't trivially predict "Normal" 100% of the time.

### Pipeline Data Flow (Universal Model Inside `unified_food_price_model.joblib`)

```
Raw Dict / DataFrame (18 input fields: 2 categorical + 16 numeric)
       │
       ▼
ColumnTransformer
  ├─ OneHotEncoder(['commodity_name', 'category'], handle_unknown='ignore') → 7 columns
  └─ StandardScaler([16 numeric features])                                 → 16 columns
       │  (Total = 23 scaled features)
       ▼
Ridge Regression (L2, alpha=1.0, Intercept = ₹2,830.72)
  ├─ Top learned weights: price_lag_7 (+1023.23), rolling_mean_7 (+721.54),
  │  price_lag_1 (+552.73), rolling_max_7 (+201.66), momentum (+178.07)
  ├─ Category intercepts: Moong/Pulses (+8.48 each), Tomato (+4.38), Onion (-6.54), Sugar (-6.32)
  └─ Predicted Price (₹/Quintal) → Converted to ₹/Kg for consumer display
```

---

## 9. Comprehensive Model Evaluation & Benchmarks (Component G — 8 Marks)

### 9.1 Evaluation Metrics Used

- **MAE (`Mean Absolute Error`):** $\frac{1}{n}\sum |y_i - \hat{y}_i|$ — linear penalty in ₹/Qtl; directly shows rupee error per 100 kg.
- **MSE & RMSE (`Root Mean Squared Error`):** $\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$ — penalizes large spike errors quadratically; RMSE $\ge$ MAE always.
- **$R^2$ (`Coefficient of Determination`):** $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ — proportion of variance explained relative to mean prediction. *(Can be negative if worse than predicting the mean!)*
- **MAPE (`Mean Absolute Percentage Error`):** $\frac{100}{n}\sum \frac{|y_i - \hat{y}_i|}{y_i}$ — scale-independent % error essential when comparing ₹700 Tomato vs. ₹7,000 Moong Dal.
- **Classification Metrics:** Accuracy, Precision ($\frac{TP}{TP+FP}$), Recall ($\frac{TP}{TP+FN}$), F1-Score, Confusion Matrix, and **ROC-AUC** (ranking ability across thresholds).
- **Clustering Metrics:** **Silhouette Score** ($[-1, +1]$, higher = tighter & better-separated clusters) and **Davies-Bouldin Index** (lower = better separation).

### 9.2 Unified Multi-Commodity Model Leaderboard (675 Unseen Test Sessions)

| Rank | Algorithm (`train_unified_model.py`) | MAE (₹/Qtl) | RMSE (₹/Qtl) | $R^2$ Score | MAPE (%) | Assessment |
|:---:|---|---:|---:|---:|---:|---|
| 1 | **Ridge Regression (L2, $\alpha=1.0$)** | **65.97** | **194.64** | **0.9935** | 6.65% | **CHAMPION** (Lowest RMSE & MAE; ₹0.66/Kg avg error) |
| 2 | **Multiple Linear Regression** | 66.38 | 194.70 | 0.9934 | **6.61%** | Virtually identical to Ridge; slightly higher RMSE variance |
| 3 | **Lasso Regression (L1, $\alpha=0.5$)** | 66.37 | 194.71 | 0.9934 | 6.62% | Sparse feature selection matches OLS |
| 4 | **ElasticNet ($\alpha=0.1, l_1=0.5$)** | 79.93 | 198.47 | 0.9932 | 8.06% | Over-shrunk due to higher $\alpha$ |
| 5 | **Gradient Boosting Regressor** | 103.86 | 209.37 | 0.9924 | 10.08% | Strong tree baseline, weaker extrapolation |
| 6 | **Random Forest Regressor** | 107.15 | 209.62 | 0.9924 | 10.64% | Struggles above training price ceilings |
| 7 | **Decision Tree Regressor (`depth=10`)** | 125.90 | 230.80 | 0.9908 | 11.88% | Step-function predictions create staircase errors |
| 8 | **Support Vector Regressor (RBF)** | 149.00 | 271.98 | 0.9872 | 14.53% | RBF struggles across discontinuous ₹700–₹7,000 plateaus |

#### Per-Commodity Breakdown of Champion Unified Ridge Pipeline (675 Test Rows)

| Commodity | Category | Test Samples | Mean Price (₹/Qtl) | Test MAE (₹/Qtl) | MAE (₹/Kg) | Test RMSE (₹/Qtl) | Test $R^2$ | Test MAPE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Moong Dal** | Pulses | 129 | 7,013.95 | **25.14** | **₹0.25** | 47.74 | **0.9833** | **0.36%** |
| **Sugar** | Sugars & Sweets | 124 | 3,222.32 | 40.98 | ₹0.41 | 177.63 | **0.0560** | **1.10%** |
| **Tomato** | Vegetables | 156 | 761.06 | 72.34 | ₹0.72 | 95.46 | **0.8384** | 10.11% |
| **Onion** | Vegetables | 266 | 909.19 | 93.68 | ₹0.94 | 273.83 | 0.4770 | 10.25% |
| **OVERALL** | **All Groups** | **675** | — | **65.97** | **₹0.66** | **194.64** | **0.9935** | **6.65%** |

*Residual Check (Notebook 05 Cell 8): Mean residual on 675 test rows = **₹1.02/Qtl** (nearly zero bias), Std Dev = **₹194.64/Qtl**.*

### 9.3 Dedicated Crop Specialist Benchmarks (`model_metadata_tomato.json` & `onion`)

- **Tomato Specialist (`156` unseen test rows from 1,039 Pune series):**
  - **Champion: Lasso Regression ($L_1$)** — MAE **₹68.66**, RMSE **₹93.35**, $R^2 = \mathbf{0.8454}$, MAPE **9.36%**.
  - **Beats Naive Persistence Baseline** ($y_{t+1} = y_t$): MAE ₹83.17, RMSE ₹112.11, $R^2 = 0.7771$, MAPE 11.25% (a **₹18.76/Qtl RMSE reduction!**).
  - *(Note: On the `873`-row exact-Pune subset tested in Notebook 05 Cell 6 (`132` test rows), Lasso still beats that subset's Naive Baseline: RMSE **₹111.45 vs ₹114.48**, $R^2$ **0.7647 vs 0.7518**.)*
  - Poor performers on Tomato test: KNN ($k=7$) RMSE ₹221.38 ($R^2=0.1307$), SVR RMSE ₹184.16, AdaBoost RMSE ₹161.72, Decision Tree RMSE ₹158.83.
- **Onion Specialist (`152` unseen test rows from `1,007` Pune series):**
  - **Champion: SVR (RBF Kernel)** — MAE **₹92.68**, RMSE **₹179.06**, $R^2 = \mathbf{0.3446}$, MAPE **9.10%**.
  - **Beats Naive Persistence Baseline:** MAE ₹97.70, RMSE ₹191.32, $R^2 = 0.2518$, MAPE 9.45%.
  - Lasso (RMSE ₹182.91), Ridge (RMSE ₹183.21), MLR (RMSE ₹183.58), and Stacking (RMSE ₹184.83) followed closely, while Decision Tree overfit badly ($R^2 = -2.3169$).

### 9.4 Optimization, Classification & Clustering Results (`reports/*.csv`)

1. **Gradient Descent Optimization (`gradient_descent_comparison_tomato.csv`, 500 epochs on test set):**
   - **Mini-batch GD (`batch=32`):** MAE **₹69.01**, RMSE **₹93.58**, $R^2 = \mathbf{0.8447}$ — **Winner** (fastest stable convergence).
   - **Batch GD:** MAE ₹69.72, RMSE **₹94.23**, $R^2 = 0.8425$ — smooth monotonic descent.
   - **Stochastic GD (SGD):** MAE ₹70.54, RMSE **₹95.23**, $R^2 = 0.8392$ — noisy oscillations on single-sample updates.
   - *(On Notebook 04's 200-epoch validation test: Mini-batch ₹124.73 < Batch ₹128.28 < SGD ₹149.77.)*
2. **Price Spike Classification (`is_price_spike`, $\pm 10\%$ threshold):**
   - **Tomato (`classification_comparison_tomato.csv`, 156 test sessions):**
     - **Gaussian Naive Bayes:** Highest **Recall = 96.49%** & **F1 = 0.5392** (Acc 39.74%, AUC 0.4618) — acts as a high-sensitivity crisis safety net.
     - **AdaBoost Classifier:** Highest **Accuracy = 52.56%** (Prec 38.36%, Recall 49.12%, AUC 0.5178).
     - **Logistic Regression:** Highest **ROC-AUC = 0.5249** (Acc 47.44%, Recall 66.67%, F1 0.4810).
   - **Onion (`classification_comparison.csv`, 152 test sessions):**
     - **Stacking Classifier:** Highest **Accuracy = 63.16%**, Precision **35.00%**, **ROC-AUC = 0.5676**.
3. **Clustering & PCA (`clustering_comparison_tomato.csv` & PCA verification):**
   - **K-Means ($k=2$ via Elbow + Silhouette):** Silhouette = **0.4747**, Davies-Bouldin = **0.9806** (separates calm vs. turbulent trading regimes).
   - **Hierarchical (Ward Linkage, $k=2$):** Silhouette = **0.4175**, Davies-Bouldin = **1.1671** (Dendrogram verification).
   - **DBSCAN ($\epsilon=2.5, \text{min\_samples}=10$):** Silhouette = **0.5190**, Davies-Bouldin = **0.7180**, isolating **2 dense clusters + 71 anomalous shock days** as noise points (`label = -1`).
   - **PCA Dimensionality Reduction:** PC1 captures **55.58%**, PC1+PC2 capture **65.71%**, and **8 Principal Components preserve 97.81% (>95%)** of variance from the 15-dimensional feature space.

---

## 10. Production Deployment, Engel's Law Engine & Ethics (Component H — 6 Marks)

### 10.1 Production Inference & UI Architecture

1. **Zero-Retraining Inference Engine ([`src/inference.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/src/inference.py)):**
   - `UnifiedCommodityPredictor` loads [`models/unified_food_price_model.joblib`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/models/unified_food_price_model.joblib) + [`unified_model_metadata.json`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/models/unified_model_metadata.json) once into memory (cached via `@st.cache_resource` in Streamlit).
   - Supports full 18-feature input OR minimal input (`commodity`, `current_price`, `arrivals`, `month`, `day_of_week`) where plausible lag heuristics and keyword category/perishability inference (`infer_category_and_perishability()`) auto-fill the feature vector—allowing unseen commodities like **Potato** or **Rice** to run smoothly via `handle_unknown='ignore'`.
   - `CommodityPricePredictor` serves the dedicated crop specialists (`Tomato` Lasso / `Onion` SVR).
2. **Streamlit Dashboard ([`app.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/app.py)) — 4 Live Modules:**
   - **Module 1 (`🔮 Commodity Price Forecaster`):** Forecasts $t+1$ price in ₹/Qtl & ₹/Kg, shift %, Surge/Crash/Stable banner, optional lag overrides, comparison bar plot.
   - **Module 2 (`🛒 Household Affordability & Engel's Law`):** Income slider, Rural/Urban selector, family size (MPCE), 5-item grocery calculator, Vulnerability status banner, metrics (Food Basket Cost, Share %, Discretionary Balance, NSSO Baseline), and Engel Curve visual gauge.
   - **Module 3 (`📊 Model Benchmarks & Syllabus Metrics`):** Interactive tabs displaying tables & saved 300 DPI figures for Universal Model, Crop Specialists, Gradient Descent, Spike Classification, and Clustering/PCA.
   - **Module 4 (`🎓 Academic Viva & Evaluation Rubric`):** 30-mark checklist + top Viva Q&As.
3. **Cloud Deployment Options (Hugging Face & Streamlit Cloud):**
   - Hugging Face Spaces moved its custom **Docker SDK** to a paid PRO tier, so while we keep a valid port-7860 [`Dockerfile`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/Dockerfile) for containerized platforms, we built a 100% free **Gradio SDK** interface ([`gradio_app.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/gradio_app.py) / [`deployment/huggingface/app.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/deployment/huggingface/app.py)) for Hugging Face Free CPU Basic (2 vCPU, 16 GB RAM) + documented free **Streamlit Community Cloud** deployment ([`deployment_guide.md`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/deployment/huggingface/deployment_guide.md)).
4. **Validation Contracts & Unit Tests:**
   - [`src/schemas.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/src/schemas.py): Pydantic v2 models enforcing `gt=0` / `ge=0` constraints for API readiness.
   - [`tests/test_affordability_engine.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/tests/test_affordability_engine.py) & [`tests/test_model_pipeline.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/tests/test_model_pipeline.py): 5 automated `pytest` checks verifying affordability math, alias keys, zero-income error handling, model loading, and inference output (`5 passed`).

### 10.2 Deterministic Household Affordability Engine & Engel's Law ([`src/affordability.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/src/affordability.py))

Why deterministic rather than ML? **Accountability and auditability.** Policymakers and families require exact arithmetic rather than an approximated regression when computing household budgets:

$$\text{Food Basket Cost} = \sum_{i=1}^k (\text{Quantity}_i \times \text{Retail Price}_i), \qquad \text{Food Share (\%)} = \frac{\text{Food Basket Cost}}{\text{Monthly Income}} \times 100$$

$$\text{Discretionary Income Remaining} = \text{Monthly Income} - \text{Food Basket Cost}$$

Calibrated against official **NSSO HCES 2022–23 Benchmarks**:
- Top 20% Fractile Avg Food Share: **29.80%** | Urban Avg: **39.17%** (MH Urban MPCE ₹6,611)
- Rural Avg Food Share: **46.38%** (MH Rural MPCE ₹4,252) | Bottom 20% Fractile: **56.60%**

| Food Budget Share | Vulnerability Classification (`classify_affordability`) |
|---|---|
| `< 30.0%` | **Low Vulnerability** *(Food Secure — Engel Curve Top Quartile)* |
| `30.0% – 45.0%` | **Moderate Vulnerability** *(Normal Middle-Income Range)* |
| `45.0% – 60.0%` | **High Vulnerability** *(Food Insecure — Resembles Bottom 30% Fractile)* |
| `> 60.0%` | **Critical Food Distress** *(Extreme Poverty / Severe Food Deprivation)* |

### 10.3 Ethical & Governance Safeguards

1. **Data Provenance & Fair Scraping:** Only public Indian government portals used; zero CAPTCHA bypassing; raw source names preserved cleanly.
2. **Socioeconomic Inclusivity:** Explicitly includes affordable essential calories/vegetables (Tomato, Onion, Sugar) alongside expensive protein (Moong Dal) and bottom-fractile NSSO benchmarks.
3. **Algorithmic Transparency:** Champion model is interpretable Ridge Regression (`coef_` inspectable feature by feature), never an opaque black box.
4. **Consumer Privacy:** Household earnings and grocery entries execute purely in-memory—zero database logging, cookies, or PII storage.
5. **Decision Support, Not Speculation:** Advisory warnings only; explicitly discloses inability to foresee sudden exogenous black-swan events (cyclones, export bans).

---

## 11. Every ML Algorithm Explained (Short & Deep Reference — 22/22 Syllabus Items + Variants)

Here is every algorithm from [`ML_algos.txt`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/ML_algos.txt) plus the additional benchmarked models (`ElasticNet`, `GradientBoostingRegressor`, `Naive Baseline`):

### A. Regression Algorithms (Continuous Next-Day Price Forecasting)

1. **Naive Persistence Baseline *(Benchmark)*:** Predicts tomorrow equals today ($\hat{y}_{t+1} = y_t$). *Role:* Mandatory zero-intelligence hurdle; any genuine ML model must beat its RMSE (**₹112.11** Tomato, **₹191.32** Onion).
2. **Simple Linear Regression (SLR):** Fits one line $\hat{y} = \beta_0 + \beta_1 x$ using only `price_lag_1`. *Project finding:* Unscaled Tomato equation $\text{Price}_{t+1} = 0.958 \cdot P_t + 56.08$ proves extreme first-order autoregression, though missing supply/volatility cues.
3. **Multiple Linear Regression (MLR):** Fits $\hat{y} = \beta_0 + \sum_{j=1}^p \beta_j x_j$ via Ordinary Least Squares (OLS) minimizing $\sum(y_i - \hat{y}_i)^2$. *Project result:* Huge leap to RMSE **₹94.08** on Tomato (`R²=0.8430`) and **₹194.70** (`R²=0.9934`) on Unified, but unregularized weights risk instability under lag collinearity.
4. **Polynomial Regression (Degree 2):** Expands features via `PolynomialFeatures(degree=2)` ($x_i^2, x_i x_j$) before fitting regularized linear regression. *Project finding:* Captures curved interactions, but expands 15 features $\to$ 135 terms (overfits beyond degree 2 due to **Curse of Dimensionality**).
5. **Ridge Regression ($L_2$ Regularization) — CHAMPION UNIFIED MODEL:** Minimizes $\sum(y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^p \beta_j^2$. Shrinks correlated feature weights proportionally toward zero (never strictly zero). *Why it won (**MAE ₹65.97, RMSE ₹194.64, R² 0.9935**):* Stabilizes correlated lags + one-hot commodity offsets simultaneously.
6. **Lasso Regression ($L_1$ Regularization) — CHAMPION TOMATO SPECIALIST:** Minimizes $\sum(y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^p |\beta_j|$. Sharp diamond $L_1$ constraint forces uninformative coefficients to **exact zero** (automatic feature selection). *Why it won on Tomato (**RMSE ₹93.35, R² 0.8454**):* Eliminated `price_lag_3`, `rolling_arrival_mean_7`, and `is_monsoon_season` cleanly.
7. **ElasticNet Regression *(Unified Benchmark)*:** Combines both penalties: $\alpha \cdot l_1 \sum |\beta_j| + \frac{\alpha(1-l_1)}{2}\sum \beta_j^2$ (`alpha=0.1, l1_ratio=0.5`). Balances grouped shrinkage ($L_2$) and sparsity ($L_1$). Achieved `R²=0.9932` on Unified.

### B. Optimization Algorithms (How Linear Weights Are Learned Iteratively)

Instead of solving OLS via matrix inversion $(X^TX)^{-1}X^Ty$, iterative optimizers walk downhill on MSE Loss $\theta_{k+1} = \theta_k - \eta \nabla L(\theta_k)$:
8. **Batch Gradient Descent:** Computes gradient over **all $N$ training rows** per epoch. *Behavior:* Smooth, deterministic, monotonic loss curve; slower on huge datasets (**Test RMSE ₹94.23/Qtl**).
9. **Stochastic Gradient Descent (SGD):** Updates weights using **1 sample** at a time (`SGDRegressor`). *Behavior:* Fast updates, jagged oscillating loss curve that helps jump shallow plateaus (**Test RMSE ₹95.23/Qtl**).
10. **Mini-batch Gradient Descent (`batch_size=32`):** Uses **32 randomly permuted rows** per `partial_fit()` step. *Behavior:* Combines matrix vectorization speed with mild noise regularization — achieved the **lowest GD Test RMSE (₹93.58/Qtl, $R^2=0.8447$)**.

### C. Classification Algorithms (Binary Price Spike Detection `|>10%|`)

11. **Logistic Regression:** Linear combination passed through sigmoid $\sigma(z) = \frac{1}{1+e^{-z}} \in (0,1)$, trained via Log-Loss (Cross-Entropy). *Result:* Highest Tomato **ROC-AUC (0.5249)** and solid 66.7% recall with `class_weight='balanced'`.
12. **Gaussian Naive Bayes:** Applies Bayes' Theorem $P(y|X) \propto P(y)\prod P(x_j|y)$ assuming conditional independence & Gaussian feature densities. *Result:* **96.49% Recall** (F1 `0.5392`) — catches almost every tomato surge at the cost of false alarms.
13. **K-Nearest Neighbors (KNN Classifier & Regressor):** Instance-based ("lazy") learner predicting by majority vote (`k=5` clf) or Euclidean distance-weighted average (`k=7` reg) of nearest standardized neighbors. *Limitation:* Degrades in 15D+ spaces and fails to extrapolate trending prices (Tomato Reg RMSE ₹221.38).
14. **Decision Tree (CART — Classification & Regression Tree):** Recursively splits feature space into orthogonal thresholds minimizing **Gini Impurity** (classification) or **MSE variance** (regression). *Pros/Cons:* Interpretable rules, prone to staircase overfitting on unseen price peaks (`max_depth` tuned).
15. **Support Vector Machine (SVC & SVR with RBF Kernel) — CHAMPION ONION SPECIALIST:**
    - **SVC** maximizes the hyperplane margin $\frac{2}{\|w\|}$ between classes after projecting points via **RBF Kernel** $K(x, x') = \exp(-\gamma\|x-x'\|^2)$.
    - **SVR** fits a regression function inside an **$\epsilon$-insensitive tube** (only errors $> \epsilon=10$ are penalized, weighted by $C=100$). Won Onion (**RMSE ₹179.06**, $R^2=0.3446$) by fitting smooth non-linear storage curves.

### D. Ensemble Learning Methods (Combining Multiple Learners)

16. **Bagging (Bootstrap Aggregating — `BaggingRegressor` / `Classifier`):** Trains $B=50$ base trees in **parallel** on random with-replacement bootstrap subsets and averages predictions, reducing **variance** ($\sigma^2/B$).
17. **Random Forest (`train_unified_model` / `train_and_evaluate` / NB04):** Bagging **plus random feature subsampling** at each node split to **decorrelate trees**, preventing `price_lag_1` from dominating every single tree.
18. **AdaBoost (Adaptive Boosting — Regressor & Classifier):** Sequential boosting assigning higher sample weights $w_i$ to previously mispredicted days so subsequent shallow trees focus on hard crisis days. Achieved **top Tomato classification Accuracy (52.56%)**.
19. **Boosting / Gradient Boosting (`GradientBoostingRegressor`):** Fits each new sequential tree directly to the **negative gradient (pseudo-residuals)** $y_i - \hat{y}_i$ of prior trees, gradually minimizing loss (**RMSE ₹209.37, $R^2=0.9924$** on Unified).
20. **Stacking (`StackingRegressor` / `Classifier`):** Level-0 heterogeneous base learners (e.g., Ridge + DT + KNN, or LR + KNN + DT) generate out-of-fold predictions (`cv=3` or `5`), which feed into a **Level-1 Meta-Learner** (Ridge/LogReg). Won **Onion spike classification (63.16% Accuracy, 0.5676 AUC)** and ranked top-4 on Tomato regression (RMSE ₹94.56).
21. **Voting (`VotingRegressor` / `VotingClassifier`):** Aggregates multiple diverse estimators without a meta-model—averaging regression outputs or summing class probabilities (`voting='soft'`).

### E. Clustering & Dimensionality Reduction (Unsupervised Learning)

22. **K-Means Clustering:** Partitions standardized market sessions into $k$ spherical centroids minimizing Within-Cluster Sum of Squares (Inertia). Tuned $k=2 \dots 8$ via **Elbow + Silhouette**; **$k=2$** cleanly separates calm vs. volatile regimes (**Silhouette 0.4747**).
23. **Hierarchical Clustering (Agglomerative, Ward Linkage):** Bottom-up clustering merging pairs that yield minimal increase in within-cluster variance, visualized as a **Dendrogram** (**Silhouette 0.4175** Tomato, **0.4622** Onion).
24. **Density-Based Clustering (DBSCAN):** Groups dense neighborhoods (within radius $\epsilon=2.5$ with $\ge 10$ points) into clusters of arbitrary shape while flagging isolated extreme market days as **noise (`-1`)**. *Project result:* Best clustering score (**Silhouette 0.5190**, **Davies-Bouldin 0.7180**) and isolated **71 anomalous volatility sessions**.
25. **PCA (Principal Component Analysis):** Orthogonal linear projection via covariance eigendecomposition transforming 15 correlated features into uncorrelated Principal Components ordered by explained variance (**PC1 = 55.58%**, **8 PCs retain 97.81%** of total variance).

---

## 12. Complete Glossary of Domain, Economic, Statistical & ML Terms

### Agricultural Economics & Policy Terms
- **APMC (Agricultural Produce Market Committee):** State-regulated wholesale agricultural market yard protecting farmers via transparent open-outcry/e-NAM auctions.
- **Mandi:** Physical agricultural wholesale market yard (e.g., Pune Gultekdi, Lasalgaon).
- **Min / Max / Modal Price:** Lowest daily bid, highest export-grade bid, and **Modal** (highest-volume clearing price — the standard inflation benchmark).
- **Quintal (Qtl):** Standard Indian wholesale weight = **100 Kilograms** (`₹1,500/Qtl` = `₹15.00/Kg`).
- **Metric Tonne:** **1,000 Kilograms** = 10 Quintals (unit for physical arrivals).
- **Kharif vs. Rabi Crop:** *Kharif* = monsoon-sown (harvested Oct–Dec); *Rabi* = winter-sown (harvested Mar–May).
- **Chal Storage:** Traditional ventilated onion storage sheds used after rabi harvest; suffer monsoon humidity rot driving Oct–Nov spikes.
- **MSP (Minimum Support Price) & FRP (Fair & Remunerative Price):** Govt-guaranteed floor price for staples (pulses/grains) and legally mandated mill price for sugarcane stabilizing Sugar CV (10.75%).
- **Rockets & Feathers Effect:** Asymmetric price transmission where consumer retail prices shoot up like a rocket on wholesale spikes but drift down slowly like a feather on wholesale gluts.
- **CPI (Consumer Price Index, Base 2012=100):** Official MoSPI index measuring percentage cost change of a fixed consumer basket (`YoY Inflation %`).
- **NSSO / HCES / MPCE / Fractiles:** National Sample Survey Office's Household Consumption Expenditure Survey (2022–23) measuring **Monthly Per Capita Consumption Expenditure** across income percentiles (**Fractiles**: e.g., bottom 0–5%, top 95–100%).
- **Engel's Law (Ernst Engel, 1857):** Empirical economic law stating that as household income decreases, the **proportion** of income spent on food increases.

### Statistics, Data Engineering & ML Terms
- **Autoregressive (AR):** Series where current/future values correlate strongly with past lags ($P_t \approx \rho P_{t-1}$).
- **Coefficient of Variation ($CV$):** $(\sigma/\mu)\times 100\%$ — dimensionless relative volatility enabling fair comparison across cheap vs. expensive commodities.
- **Skewness & Kurtosis:** Skewness measures tail asymmetry ($>0$ = occasional huge upward spikes). Kurtosis measures tail heaviness (**Leptokurtic** = fat tails $\implies$ frequent extreme shocks).
- **IQR & Tukey Fence:** $\text{IQR} = Q_3 - Q_1$; statistical outliers sit beyond $Q_3 + 1.5\times\text{IQR}$.
- **Multicollinearity:** High correlation between independent predictors (`price_lag_1` & `rolling_mean_7` at $r>0.95$); destabilizes unregularized OLS weights, cured by Ridge/Lasso.
- **Lookahead Data Leakage:** Accidental use of future info ($t+1 \dots$) during training or scaling $\implies$ fake lab accuracy that fails live.
- **Gauss-Markov & Homoscedasticity:** Regression ideal where residuals have zero mean and constant spread across predicted values (absence of funnel-shaped heteroscedasticity).
- **Bias-Variance Tradeoff:** Total Error $= \text{Bias}^2 + \text{Variance} + \sigma_{\epsilon}^2$. Too simple $\to$ High Bias (Underfitting); too complex (Deg 3 Poly, deep Tree) $\to$ High Variance (Overfitting).
- **Regularization ($L_1$ vs $L_2$):** Penalizing large model weights to curb variance. $L_2$ (Ridge) shrinks weights smoothly; $L_1$ (Lasso) forces redundant weights to exact zero.
- **Curse of Dimensionality:** Rapid feature explosion ($15 \to 135 \to 800+$ polynomial terms) making fixed sample counts sparse and causing distances (KNN) & OLS fits to degrade.
- **`ColumnTransformer` & `Pipeline`:** Scikit-learn wrappers chaining column-specific preprocessing (`OneHotEncoder` + `StandardScaler`) directly with the estimator inside a single atomic object—preventing inference schema drift and CV leakage.
- **`handle_unknown='ignore'`:** Encoder setting ensuring unseen categories (like Potato or Rice) map cleanly to an all-zero dummy vector instead of raising `ValueError`.
- **Kernel Trick (RBF):** Implicitly mapping non-linear inputs into higher-dimensional space using Radial Basis Functions so linear separators/tubes fit curved relationships.
- **Confusion Matrix ($TP, FP, TN, FN$):** Table of predicted vs. actual classes; **Recall** catches real crises ($TP/(TP+FN)$), **Precision** avoids false alarms ($TP/(TP+FP)$).
- **Joblib & Pydantic v2:** `joblib` serializes NumPy-heavy `scikit-learn` pipelines efficiently; `Pydantic` validates strict REST input types/bounds (`gt=0`) before inference.

---

## 13. File-by-File Codebase Map & Notebook Walkthrough Guide

### Jupyter Notebooks (Pre-Executed with Rendered Outputs)

| Notebook | Covers Rubric | Key Executed Walkthrough for Faculty |
|---|---|---|
| [`01_data_collection_overview.ipynb`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/notebooks/01_data_collection_overview.ipynb) | **A & B** | Inspects all **5 processed CSVs (189,413 rows)** + `model_ready_unified_commodities.csv` (`4,480` rows), displays multi-commodity stats, arrival tonnage, retail markups, CPI, HCES fractiles. |
| [`02_exploratory_data_analysis.ipynb`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/notebooks/02_exploratory_data_analysis.ipynb) | **B & C** | Computes **7+ descriptive stats** across Tomato, Onion, Moong Dal, Sugar; renders **5 visualizations** (timeseries, KDE histograms, Tukey boxplots, crop-calendar seasonality, correlation heatmap). |
| [`03_data_preprocessing.ipynb`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/notebooks/03_data_preprocessing.ipynb) | **D & E** | Verifies anomalies/duplicates removed (`168` dupes), builds zero-leakage `engineer_commodity_features()`, verifies chronological 70/15/15 split timelines graphically across all 4 commodities. |
| [`04_regression_model_development.ipynb`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/notebooks/04_regression_model_development.ipynb) | **F** | Trains Baseline + **13 regressors** on validation set, extracts `SLR` equation, benchmarks **Batch/SGD/Mini-batch GD** loss curves over 200 epochs, and trains the **Unified Cross-Commodity Pipeline**. |
| [`05_model_evaluation.ipynb`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/notebooks/05_model_evaluation.ipynb) | **G & H** | Evaluates **Unified Ridge** on 675 unseen test rows (`R²=0.9935`), breaks down all 4 commodities, verifies Tomato & Onion specialists beat Baseline, checks residuals (+₹1.02 mean), shows **9 classifiers**, **Clustering/PCA**, and simulates Urban ₹30k vs ₹12k household affordability shocks. |

### Repository Tree (`scripts/`, `src/`, `models/`, `tests/`)

```
ML_Project/
├── app.py                              # Main Streamlit 4-module web application
├── gradio_app.py                       # Free Gradio web application (Hugging Face Spaces SDK)
├── Dockerfile                          # Container build exposing Streamlit on port 7860
├── requirements.txt                    # Python dependencies
├── Problem_statement.txt / rubrics.md / ML_algos.txt
├── PRESENTATION_SCRIPT.md              # This complete mastery guide
│
├── data/
│   ├── raw/                            # Immutable API & scraped raw dumps + HCES PDF
│   ├── processed/                      # 5 core CSVs (189k rows) + 4 model_ready CSVs
│   └── metadata/                       # Data dictionary, API specs, collection logs
│
├── notebooks/                          # 5 pre-rendered Jupyter Notebooks (01–05)
├── models/                             # Serialized joblib pipelines & JSON metadata contracts
│   ├── unified_food_price_model.joblib # Champion Universal Ridge L2 Pipeline
│   ├── unified_model_metadata.json     # 675-test metrics & feature schema
│   ├── price_prediction_model_tomato.joblib  # Tomato Specialist (Lasso L1)
│   └── price_prediction_model_onion.joblib   # Onion Specialist (SVR RBF) + default symlink copy
│
├── scripts/
│   ├── main.py                         # Master CLI orchestrator (--all/--collect-api/--scrape/--clean/--validate/--summary)
│   ├── api/                            # mospi_client.py, data_gov_client.py, ceda_client.py
│   ├── scraping/relevant_scrapers.py   # DCA scraper, HCES PDF parser, mandi harvesters
│   ├── processing/                     # clean_data.py, normalize_data.py, validate_data.py
│   ├── generate_eda.py                 # Generates EDA stats & Figures 01–05
│   ├── build_features.py               # Multi-commodity feature engineer (--unified / --all)
│   ├── preprocess_and_prepare_data.py  # Original single-yard Pune Tomato preprocessor (873 rows)
│   ├── train_and_evaluate.py           # Early Tomato CV & hyperparameter grid search script
│   ├── train_all_models.py             # Trains ALL 22 syllabus items (29+ runs) for any crop
│   ├── train_unified_model.py          # Trains Champion Unified Multi-Commodity Pipeline
│   └── test_unified_model.py / test_model.py / inference_test.py  # Production CLI verification tools
│
├── src/
│   ├── inference.py                    # UnifiedCommodityPredictor & CommodityPricePredictor
│   ├── affordability.py                # Deterministic NSSO Engel's Law calculator
│   └── schemas.py                      # Pydantic v2 validation schemas
│
├── reports/                            # Audit docs, viva guide, CSV comparisons, 16 PNGs in figures/
├── tests/                              # pytest suite (test_affordability_engine.py, test_model_pipeline.py)
└── deployment/huggingface/             # HF Gradio app.py, README.md, deployment_guide.md
```

---

## 14. Hard Faculty Viva Questions & Model Answers (With Honest Tradeoffs)

### Q1: Why is `R²` 0.9935 on the Unified Model while Sugar's per-commodity `R²` is only 0.0560 and Onion is 0.4770? Does `0.0560` mean Sugar failed?
**Answer:** No—this is a classic statistical scaling effect of the $R^2 = 1 - \frac{\text{SS}_{res}}{\text{SS}_{tot}}$ formula!
- In the **Unified test set (675 rows)**, prices span from ~₹200/Qtl (Tomato) to ~₹9,500/Qtl (Moong Dal), making total variance $\text{SS}_{tot}$ enormous. Because our one-hot encoded Ridge model nails each commodity's price level, global $R^2$ is **0.9935**.
- Inside **Sugar alone**, government FRP mandates and mill quotas keep Sugar almost flat within ~₹200/Qtl across the test set, making $\text{SS}_{tot}$ tiny. Any minor daily fluctuation yields a low $R^2$, **yet Sugar's MAPE is an outstanding 1.10% (only ₹40.98/Qtl or ₹0.41/Kg error!)**. That proves why you must pair $R^2$ with **MAPE and MAE**.

### Q2: Why did Linear Regularized Models (Ridge / Lasso) beat Random Forest and Gradient Boosting on your Unified Dataset?
**Answer:** Two domain reasons:
1. **Linear Autoregressive Dominance:** Next-session wholesale clearing price strongly follows $P_{t+1} \approx w_1 P_t + w_2 \bar{P}_7 + \dots$, which linear models represent natively in one hyperplane without stair-stepping.
2. **Tree Extrapolation Failure:** Decision Trees, Random Forests, and Gradient Boosting partition training ranges into axis-aligned leaves capped at historical leaf averages. Whenever test prices drift above or below training extremes (or into unseen price combinations), trees clip flatly, whereas Ridge/Lasso extrapolate smoothly.

### Q3: Why did Lasso win Tomato Specialist while Ridge won the Unified Model and SVR won Onion?
**Answer:** Each regularizer matches a different data geometry:
- **Tomato Specialist (`15` numerical features):** Has noisy redundant features (e.g., `price_lag_3`, 7-day average arrivals). **Lasso ($L_1$)** zeroed those three out cleanly to lower test RMSE to **₹93.35**.
- **Unified Model (`23` encoded columns):** Contains grouped one-hot dummies alongside collinear lags. You don't want $L_1$ dropping one commodity indicator arbitrarily; **$L_2$ (Ridge)** distributes shrinkage smoothly across groups (**MAE ₹65.97**).
- **Onion Specialist:** Storage releases from Nashik/Pune *chals* create non-linear inflection curves that an **RBF-kernel SVR ($\epsilon=10, C=100$)** captured better than straight hyperplanes (**RMSE ₹179.06 vs Baseline ₹191.32**).

### Q4: Your spike classifiers only get 47–63% Accuracy and 0.52–0.57 ROC-AUC. Why aren't they 90% accurate?
**Answer:** Be completely honest—this shows scientific maturity! Predicting an **exact single-day $\pm 10\%$ auction shock** strictly from historical prices and arrivals is fundamentally constrained by **unobserved exogenous shocks** (sudden overnight highway flooding, unexpected export bans, sudden buyer strikes at Gultekdi mandi). Even so:
- **Gaussian Naive Bayes achieved 96.49% Recall** on Tomato spikes—serving as an asymmetric "high-recall early warning alarm" (much safer to issue a caution than miss a crisis!).
- **Stacking Classifier reached 63.16% Accuracy (`0.5676 ROC-AUC`)** on Onion spikes.
- Adding weather rainfall/temperature APIs and policy event flags would boost spike separability further.

### Q5: How do you prevent data leakage in BOTH feature engineering AND scaling/splitting?
**Answer:** Three guarantees:
1. All rolling features use `.shift(1).rolling(...)` so the target day $t+1$ (and even day $t$ in unified lag features) never contaminates rolling statistics.
2. Splitting is strictly **chronological (70/15/15)** per commodity before pooling, plus expanding-window **`TimeSeriesSplit`** during CV—never random shuffling.
3. **`StandardScaler`** inside scikit-learn `Pipeline`/`ColumnTransformer` computes $\mu_{\text{train}}, \sigma_{\text{train}}$ solely from Training folds before transforming Validation/Test sets.

### Q6: If `UnifiedCommodityPredictor` was trained only on Tomato, Onion, Moong Dal, and Sugar, what happens if someone enters "Potato" or "Rice" in the Streamlit app?
**Answer:** Because `OneHotEncoder(handle_unknown='ignore')` is inside our `ColumnTransformer` (`scripts/train_unified_model.py`), unseen strings do **not** throw a `ValueError`. Instead, `infer_category_and_perishability()` assigns `"Cereals"` / `"Vegetables"` + perishability (`0` or `1`), and any unknown commodity dummy simply encodes as an all-zero vector while `Ridge` predicts seamlessly from the 16 standardized numerical price/supply/calendar features!

### Q7: Why did you check `model_ready_pune_tomato.csv` (`873` rows) vs `1,039` rows and `3,755` vs `4,480` rows across scripts and notebooks? *(Bonus Deep Mastery Question!)*
**Answer:** Notice the difference in market granularity:
- [`scripts/preprocess_and_prepare_data.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/preprocess_and_prepare_data.py) filters exact `market_name == "Pune"` (`873` rows: `611/130/132`) with `price_lag_1 = P(t)`.
- [`scripts/build_features.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/build_features.py) aggregates all Pune sub-markets (`Pune` + `Pimpri` + `Hadapsar` + `Khadiki`) via `.str.contains('Pune')` yielding **`1,039` Tomato rows (`156` test)**, keeps Pune (`1,007`) and Lasalgaon (`758`) Onion separate, and pools them with Moong Dal (`854`) and Sugar (`822`) into **`4,480` total unified rows** using strictly shifted `.shift(1)` lags. Notebook 03 demonstrates pooling both Onion hubs onto one date axis (`3,755` rows), while production uses the full `4,480`-row unified store (`unified_food_price_model.joblib`).

### Q8: Why isn't the Household Affordability Calculator itself a trained Neural Network or Regression Model?
**Answer:** Grocery budgeting is governed by exact arithmetic and **Engel's Law**, not noisy statistical estimation. A citizen or policy auditor must verify `20 kg × ₹38 + ... = ₹1,472`, divide by ₹25,000 income (`5.89%`), and compare deterministically against NSSO HCES 2022–23 thresholds (`39.17%` Urban / `46.38%` Rural / `56.60%` Bottom 20%). ML solves the *uncertain future* (commodity prices $P_{t+1}$); deterministic econometrics solves the *exact budget consequence*.

### Q9: Why did you build both `app.py` (Streamlit) and `gradio_app.py` (Gradio) plus `Dockerfile`?
**Answer:** Hugging Face Spaces recently moved custom **Docker SDK** containers (used to host Streamlit in `Dockerfile` on port 7860) behind a paid PRO plan. Rather than paying or dropping cloud deployment, we preserved full local/Streamlit Cloud support via `app.py` AND engineered a native 100% free **Gradio SDK** clone (`gradio_app.py` / `deployment/huggingface/app.py`) sharing the exact same `src/inference.py` and `src/affordability.py` backend modules so it hosts free on HF CPU Basic (2 vCPU, 16 GB RAM).

### Q10: How did you verify reliability in software engineering terms?
**Answer:**
1. Decoupled training (`scripts/train_*.py`) from serving (`src/inference.py`) using `joblib` + JSON feature contracts so inference takes milliseconds with zero retraining.
2. Verified both prediction keys and edge cases (e.g., `monthly_income <= 0` raises `ValueError`, unified alias keys prevent UI crashes like the fixed `KeyError: 'affordability_status'`) using an automated `pytest` suite (`tests/` — 5/5 passing).
3. Enforced clean project structure and PII-free execution.

---

> **Quick Checklist Before Walking Into Viva:**
> 1. Launch Streamlit locally right before presenting: `py -3.12 -m streamlit run app.py`
> 2. Keep Notebooks `01`–`05` open in IDE/Jupyter with all pre-rendered plots ready to scroll.
> 3. Remember the 4 killer headline stats:
>    - **Corpus:** `189,413` government records (5 datasets); Unified store = `4,480` rows (`3,133 / 672 / 675` chronological split).
>    - **Unified Champion:** Ridge ($L_2$) **MAE ₹65.97/Qtl (~₹0.66/Kg)**, **RMSE ₹194.64/Qtl**, **$R^2 = 0.9935$**, **MAPE 6.65%** (Moong Dal **0.36%**, Sugar **1.10%**, Tomato **10.11%**, Onion **10.25%**).
>    - **Specialists vs Persistence Baseline:** Tomato Lasso **RMSE ₹93.35 vs ₹112.11** ($R^2=0.8454$); Onion SVR **RMSE ₹179.06 vs ₹191.32** ($R^2=0.3446$).
>    - **NSSO Engel Benchmarks:** Urban **39.17%**, Rural **46.38%**, Bottom 20% **56.60%** food budget share.
