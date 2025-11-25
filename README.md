# UK Road Traffic Accident Analysis (2020)

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)]()
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification%20%7C%20Clustering-orange.svg)]()
[![Status](https://img.shields.io/badge/Project-Completed-brightgreen.svg)]()

This repository contains an end-to-end data analysis and machine learning project exploring **Road Traffic Accidents (RTAs) in Great Britain during 2020**. The work combines **SQL**, **Python**, **exploratory data analysis**, **clustering**, **association rule mining**, and **classification models** to uncover patterns in accident occurrence and severity.:contentReference[oaicite:0]{index=0}

---

## 1. Project Overview

Road traffic accidents are a major public health and economic burden due to congestion, fatalities, injuries, and infrastructure damage. This project analyses a relational database of UK accident records with four key tables:

- `Accident`
- `Vehicle`
- `Casualty`
- `Lsoa`

The database covers **2017–2020**, but this analysis focuses specifically on **accidents that occurred in 2020**.:contentReference[oaicite:1]{index=1}

The aims are to:

- Understand **when** and **where** accidents occur most frequently  
- Characterise patterns involving **motorcycles** and **pedestrians**  
- Identify **road and environmental factors** that influence **accident severity**  
- Build **predictive models** for accident outcomes (fatal vs non-fatal)  
- Provide **data-driven recommendations** for road safety policy and planning

All analysis is implemented in a Jupyter Notebook, with interpretation and visualisations documented in a structured PDF report.

---

## 2. Project Highlights

- ⏰ **Temporal patterns**: Accidents peak around **17:00 (5pm)** and are most frequent on **Fridays**, aligning with evening rush hour.:contentReference[oaicite:2]{index=2}  
- 🏍️ **Motorcycle risk**:
  - Motorcycles **≤ 125cc** account for approximately **61%** of all motorcycle accidents.
  - Across engine capacities, accidents cluster around **17:00**, with strong weekday peaks (especially Fridays).:contentReference[oaicite:3]{index=3}  
- 🚶 **Pedestrians**: Pedestrian-involved accidents peak around **15:00**, with Fridays again showing the highest counts.:contentReference[oaicite:4]{index=4}  
- 🌍 **Location & context**:
  - `speed_limit` and `urban_or_rural_area` emerge as key drivers of `accident_severity`.
  - Association rule mining confirms that **urban areas + 30 mph roads** are strongly associated with **non-fatal accidents**.:contentReference[oaicite:5]{index=5}  
- 📍 **Regional clustering (Humberside)**:
  - Clustering highlights **Kingston Upon Hull** as a major hotspot, with accident concentrations along roads such as **Anlaby Road**, **Hessle Road**, and **Spring Bank**.:contentReference[oaicite:6]{index=6}  
- 🤖 **Machine learning performance**:
  - Random Forest, Decision Tree, Gradient Boosting, and KNN are compared for predicting fatal injuries.
  - **Random Forest** achieves the best test accuracy (~**0.86**) but exhibits training accuracy of 1.0, indicating potential overfitting and scope for further regularisation.:contentReference[oaicite:7]{index=7}  

---

## 3. Methods & Techniques

### 3.1 Data Extraction & Cleaning

- Data sourced from a **SQL database** containing accident, vehicle, casualty, and LSOA tables.  
- Records filtered to include only the **year 2020**.  
- **Missing value handling**:
  - Continuous geographic variables (`location_easting_osgr`, `location_northing_osgr`, `longitude`, `latitude`) imputed using the **median** to respect skewed distributions.:contentReference[oaicite:8]{index=8}  
  - Sentinel values (e.g. `-1`) converted to `NaN`, then imputed using the **mode** per column to avoid unrealistic values and preserve distribution shape.:contentReference[oaicite:9]{index=9}  
- Numerous **helper functions** implemented to improve code structure, reusability, and readability.

### 3.2 Outlier Detection

- **Isolation Forest** applied to `longitude` and `latitude` to detect spatial anomalies in a large, skewed dataset, identifying ~0.25% of points as outliers but retaining them as genuine locations.:contentReference[oaicite:10]{index=10}  
- **IQR-based** outlier detection for `age_of_vehicle`, `age_of_driver`, and `age_of_casualty`, with outliers retained due to limited impact on aggregated statistics and their relevance as real-world extreme cases.:contentReference[oaicite:11]{index=11}  

### 3.3 Exploratory Data Analysis (EDA)

- Temporal analyses:
  - Accident counts by **hour of day** and **day of week**.
  - Dedicated breakdowns for **motorcycle categories** (≤125cc, >125cc, up to 500cc) and **pedestrian involvement**.:contentReference[oaicite:12]{index=12}  
- Visualisation stack: **Matplotlib** and **Seaborn** for statistical plots; **heatmaps** for correlation analysis; potentially **Folium** for geographic mapping in extended versions.

### 3.4 Feature Selection & Association Rule Mining

- **SelectKBest** used to select the most informative predictors for `accident_severity`. High-scoring features include:  
  - `urban_or_rural_area`  
  - `speed_limit`  
  - `second_road_class`  
  - `junction_location`:contentReference[oaicite:13]{index=13}  
- **Apriori** and `association_rules` (via `mlxtend`) applied to one-hot encoded factors to derive interpretable patterns, including rules linking:
  - Urban areas (`urban_or_rural_area_1`)
  - 30 mph limits (`speed_limit_30`)
  - Non-fatal accidents (`accident_severity_2`) with high confidence and conviction.:contentReference[oaicite:14]{index=14}  

### 3.5 Clustering (Humberside Region)

- Multiple clustering algorithms evaluated:
  - **KMeans**
  - **KMedoids**
  - **DBSCAN**
  - **Agglomerative Clustering**  
- **KMeans** achieves the best **Silhouette score (~0.62)**, outperforming alternatives (Agglomerative: 0.59, KMedoids: 0.43, DBSCAN: 0.34).:contentReference[oaicite:15]{index=15}  
- Resulting clusters visualised geographically to highlight high-risk areas, especially within **Kingston Upon Hull**.:contentReference[oaicite:16]{index=16}  

### 3.6 Classification Models

- Target: **predict fatal vs non-fatal accident outcomes**.  
- Models implemented:
  - Random Forest
  - Decision Tree
  - Gradient Boosting
  - K-Nearest Neighbours  
- Experimental setup:
  - `train_test_split` with 80/20 split.
  - Model comparison by training and test accuracy, with Random Forest emerging as best-performing but mildly overfitted.:contentReference[oaicite:17]{index=17}  
- The codebase can easily be extended to include:
  - Precision, recall, F1-score, ROC-AUC.
  - Class rebalancing (e.g. `SMOTE`, oversampling/undersampling).
  - Hyperparameter tuning via `GridSearchCV` / `RandomizedSearchCV`.

---

## 4. Repository Structure

```text
uk-road-accident-analysis-2020/
│
├── notebooks/
│   └── accident_analysis_2020.ipynb       # Main Jupyter Notebook analysis (renamed as needed)
│
|
│
├── src/                                   # (Optional) Utility scripts if refactored from notebook
│   └── ...
│
├── README.md                              # Project overview (this file)
└── requirements.txt                       # Python dependencies
