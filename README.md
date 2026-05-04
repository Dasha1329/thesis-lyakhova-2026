# Customer Value Deterioration Modelling in E-Commerce

**Bachelor's/Master's Thesis** · Daria Lyakhova · 2026

---

## Overview

This repository contains the analytical code for a thesis on modelling customer value deterioration in a Russian children's goods e-commerce retailer. The study covers the full machine learning pipeline: from raw transactional data engineering to an operational early warning system.

**Dataset:** 4,122,221 order-line records from a single retailer, calendar year 2017 (1C ERP exports).  
**Customer base:** 82,400 unique customers after preprocessing, aggregated into a quarterly client–feature matrix (82,400 records × 36 features).

---

## Research Question

> Can individual customer characteristics, observed within a single quarter, predict whether that customer will decline in commercial value in the following quarter — and at what expected speed?

---

## Methodology

The framework consists of five analytical layers:

1. **Data Engineering** — consolidation of bimonthly CSV files (Windows-1251), deduplication, financial column parsing, construction of a quarterly client–feature matrix with 36 behavioural and financial features per customer–quarter observation.

2. **Unsupervised Clustering** — systematic comparison of five algorithms (K-Means, GMM, DBSCAN, HDBSCAN with UMAP, hierarchical Ward linkage) on the 36-dimensional feature space. K-Means with k=10 selected as primary; cluster selection validated via SSE elbow, Silhouette, Davies-Bouldin, Calinski-Harabasz.

3. **RFM Segmentation & Transition Modelling** — rule-based RFM framework producing six ordered segments (Lost → Champions). First-order Markov transition model with Bayesian Dirichlet-Multinomial uncertainty quantification.

4. **Binary Downgrade Classification** — two parallel pipelines (RFM-based and K-Means-based). Six classifiers evaluated: LightGBM, Random Forest, XGBoost, MLP, Logistic Regression, SVM. LightGBM selected as primary operational model. Best RFM pipeline ROC-AUC: **0.758**; best K-Means pipeline ROC-AUC: **0.883**.

5. **Early Warning System** — four-tier risk classification with within-segment score normalisation, expected margin loss estimation, and SHAP-based feature attribution via TreeExplainer.

---

## Repository Structure

```
├── eda.ipynb                   # Exploratory data analysis and feature matrix construction
├── clustering.ipynb            # Clustering algorithm comparison (K-Means, GMM, DBSCAN, HDBSCAN, Ward)
├── overflows_rfm.ipynb         # RFM segmentation, Markov model, classifiers, survival analysis
├── overflows_rfm_new.ipynb     # Updated RFM pipeline (Q2→Q3 evaluation period)
├── overflows k-means.ipynb     # K-Means pipeline
├── data/
│   ├── client_pivot.csv        # Customer-level aggregated features
│   └── client_pivot_quarter.csv  # Quarterly customer–feature matrix (main modelling input)
```

---

## Key Results

| Pipeline | Best Model | ROC-AUC | F1 | Recall | Precision |
|---|---|---|---|---|---|
| RFM | LightGBM / Random Forest | 0.758 | 0.715 | 0.942 | 0.676 |
| K-Means | LightGBM | 0.883 | 0.733 | 0.851 | 0.744 |

- Optimal decision thresholds for tree-based ensembles: **0.29–0.32** (asymmetric cost structure: missed downgrade >> false positive)
- At Risk segment acts as systemic attractor: transition probability into At Risk ranges **0.20–0.31** regardless of source segment
- K-Means clusters correspond financially to RFM segment profiles, confirming that RFM dimensions reflect genuine discontinuities in the customer population

---

## Data

Raw transactional data (2.2 GB) is not included in this repository due to size constraints. The preprocessed quarterly feature matrix (`data/client_pivot_quarter.csv`, 65 MB) is provided as the primary modelling input.
