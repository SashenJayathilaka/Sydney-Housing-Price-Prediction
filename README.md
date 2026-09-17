# Sydney Housing Price Prediction

A machine learning mini-project that predicts residential sale prices across three
structurally different Sydney housing markets — **Mosman** (prestige, harbourside),
**Parramatta** (high-density growth corridor), and **Blacktown** (affordable outer
suburb) — and packages the result as a decision-support web app.

Built for SIT720 (Machine Learning), Deakin University.

## Overview

165 sold-property listings were manually collected from Domain.com.au (56 Mosman /
56 Parramatta / 53 Blacktown, sales from 30 May – 14 Sep 2026), cleaned, and used to
train and compare three regression models. The best-performing model (XGBoost) was
then deployed behind a Streamlit app that returns a price estimate from a property's
suburb, type, bedrooms/bathrooms/car spaces, distance to the CBD, and amenities.

**Headline result:**

| Model             | MAE          | RMSE         | R²       |
| ----------------- | ------------ | ------------ | -------- |
| Linear Regression | $771,868     | $1,132,877   | −0.40    |
| Random Forest     | $352,552     | $775,161     | 0.35     |
| **XGBoost**       | **$332,800** | **$711,388** | **0.79** |

XGBoost was the strongest fit by a wide margin, reflecting the non-linear,
suburb-conditional way price behaves across this dataset (see the full writeup in
`report/` for why). The project also compares the model's predictions against an
LLM estimate and a manual human estimate on 10 held-out properties — full results
and discussion are in Part 5 of the report.

## Repository structure

```
sydney-housing-project/
├── data/
│   ├── sydney_housing_ml_ready_clean.csv
│   ├── sold_listings_mosman_parramatta_blacktown_CLEANED.xlsx
│   └── part5_human_estimates.csv
├── notebooks/
│   └── sydney_housing_project.ipynb
│
├── app/
│   ├── app.py
│   └── trained_model.joblib
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone <this-repo-url>
cd sydney-housing-project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Reproduce the analysis

```bash
jupyter notebook notebooks/sydney_housing_project.ipynb
```

Run top to bottom. The notebook loads `data/sydney_housing_ml_ready_clean.csv`,
walks through EDA and feature engineering, trains and cross-validates all three
models, investigates the largest prediction errors, runs the Part 5 human/LLM/ML
comparison, and finishes by saving the fitted pipeline to `app/trained_model.joblib`.

### Run the app

Needs `app/trained_model.joblib` to exist first (produced by the notebook's last
cell, already included in this repo).

```bash
cd app
streamlit run app.py
```

Open the local URL Streamlit prints (typically `http://localhost:8501`), fill in a
property's details, and click **Predict price**.

## Data notes

- **Target:** sale price. **Features:** suburb, property type, bedrooms, bathrooms,
  car spaces, distance to CBD, plus 24 engineered `feat_*` amenity flags parsed from
  agent descriptions (9 standalone high-support flags + grouped composites).
- **Deliberately excluded:** land size (only ~56% populated) and floor area/year
  built (almost never populated in listings) — including them would have meant
  losing ~44% of rows or imputing values that were never observed. Bedrooms,
  bathrooms, car spaces and property type act as size proxies instead.
- **Full cleaning log:** every merge/grouping/exclusion decision is documented in
  the `Notes` sheet of `sold_listings_mosman_parramatta_blacktown_CLEANED.xlsx`.
- **Known limitation:** all sales fall within a ~3.5-month window, which supports
  cross-suburb comparison but not seasonality or year-over-year trend analysis.

## Limitations

This is a small (165-row), single-window, three-suburb academic dataset — it is a
decision-_support_ prototype, not a valuation tool. It should not be used for real
lending, insurance, or valuation decisions. See the report's critical reflection
for a full discussion of accuracy limits (particularly at the high-value extremes
of the Mosman market) and the ethical considerations of suburb-level pricing models.
