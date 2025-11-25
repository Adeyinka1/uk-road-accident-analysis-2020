# UK Road Traffic Accident Analysis (2020)

This repository contains an end-to-end data analysis and machine learning project exploring **Road Traffic Accidents (RTAs) in Great Britain during 2020**. The work combines **SQL, Python, exploratory data analysis, clustering, association rule mining, and classification models** to uncover patterns in accident occurrence and severity.:contentReference[oaicite:0]{index=0}

---

## 1. Project Overview

Road traffic accidents remain a major public health and economic concern due to congestion, fatalities, injuries, and infrastructure damage. This project analyses accident records from 2020 with the goal of:

- Understanding **when** and **where** accidents occur most frequently  
- Characterising accident patterns involving **motorcycles** and **pedestrians**  
- Identifying **environmental and road factors** associated with accident severity  
- Building **predictive models** for fatal vs non-fatal accidents  
- Providing **evidence-based recommendations** for road safety policy

The analysis uses a database of four tables (`Accident`, `Vehicle`, `Casualty`, `Lsoa`) covering UK accident records from 2017–2020, with this project focusing on **2020 only**. Data is initially extracted via SQL and further processed and modelled in Python.:contentReference[oaicite:1]{index=1}

---

## 2. Key Findings (High-Level)

Some of the main insights from the analysis and modelling are:

- **Peak accident time:** Around **17:00 (5pm)**, corresponding to evening rush hour.  
- **Most accident-prone day:** **Friday** shows the highest accident frequency overall.  
- **Motorcycle accidents:**  
  - Motorcycles **≤ 125cc** account for ~**61%** of motorcycle-related accidents.  
  - Across all motorcycle categories, accidents cluster heavily around **17:00**, with noticeable weekday peaks.:contentReference[oaicite:2]{index=2}  
- **Pedestrian-involved accidents:** Highest around **15:00**, again with **Fridays** particularly prominent.:contentReference[oaicite:3]{index=3}  
- **Important risk factors for severity:**  
  - `speed_limit` and `urban_or_rural_area` are consistently among the strongest predictors of `accident_severity`.  
  - Association rule mining (Apriori) reveals strong rules linking **urban areas + 30mph roads** with **non-fatal accident outcomes**, aligning with correlation analysis.:contentReference[oaicite:4]{index=4}  
- **Regional clustering (Humberside focus):**  
  - Clustering on Humberside regions shows **Kingston Upon Hull** as a clear hotspot for accidents, with specific major roads (e.g. Anlaby Road, Hessle Road, Spring Bank) contributing heavily.:contentReference[oaicite:5]{index=5}  
- **Model performance:**  
  - Among Random Forest, Decision Tree, Gradient Boosting, and K-Nearest Neighbours, the **Random Forest classifier** achieves the best test accuracy (~**0.86**), albeit with a training accuracy of 1.0, indicating possible overfitting.:contentReference[oaicite:6]{index=6}  

---

## 3. Methods & Techniques

### 3.1 Data Processing & Cleaning

- Data loaded from an **SQL database** with four relational tables.  
- Year filter applied to keep only **2020** records.  
- Missing values handled with:
  - **Median imputation** for skewed continuous variables (e.g. `longitude`, `latitude`).  
  - Replacement of sentinel codes (e.g. `-1`) with proper missing values and imputation using **mode** for categorical features.:contentReference[oaicite:7]{index=7}  
- Multiple helper **functions** are defined to streamline plotting and cleaning tasks.

### 3.2 Outlier Detection

- **Isolation Forest** used on geospatial coordinates (`longitude`, `latitude`) to detect sparse anomalies while handling skewed distributions.  
- **IQR-based** methods used for `age_of_vehicle`, `age_of_driver`, and `age_of_casualty`.  
- Outliers are retained because they represent valid but rare accident configurations and have limited impact on overall summary statistics.:contentReference[oaicite:8]{index=8}  

### 3.3 Exploratory Data Analysis (EDA)

- Temporal patterns: count plots by **hour of day** and **day of week**, including breakdowns for:
  - **All accidents**
  - **Motorcycles by engine capacity (≤125cc, >125cc, up to 500cc)**
  - **Pedestrian-involved accidents**  
- Visualisation stack: **Matplotlib** and **Seaborn** for histograms, bar charts, and heatmaps; **Folium** for interactive geographic mapping.

### 3.4 Feature Selection & Association Rules

- **SelectKBest** (ANOVA / chi-square) is used to select the most informative features with respect to `accident_severity`.  
- Top factors include: `urban_or_rural_area`, `speed_limit`, `second_road_class`, `junction_location`.  
- **Apriori + association_rules** (via `mlxtend`) applied to the selected features to discover interpretable patterns linking road conditions and accident severity outcomes.:contentReference[oaicite:9]{index=9}  

### 3.5 Clustering

- Comparative clustering on Humberside regions using:
  - **KMeans**
  - **KMedoids** (from `sklearn_extra`)
  - **DBSCAN**
  - **Agglomerative Clustering`**  
- **KMeans** achieves the best **Silhouette score (~0.62)** and is used to visualise accident clusters on a map, highlighting key hot-spots in Kingston Upon Hull and surrounding areas.:contentReference[oaicite:10]{index=10}  

### 3.6 Classification Modelling

- Models trained to predict **fatal vs non-fatal accidents**:
  - Random Forest
  - Decision Tree
  - Gradient Boosting
  - K-Nearest Neighbours  
- Data split using `train_test_split` (80/20).  
- Performance assessed using accuracy (and can be extended to precision/recall/F1/AUC in future work).  
- **Random Forest** yields the strongest performance but requires regularisation / tuning to mitigate overfitting.:contentReference[oaicite:11]{index=11}  

---

## 4. Repository Structure

```text
uk-road-accident-analysis-2020/
│
├── notebooks/
│   └── accident_analysis.ipynb               # Full code and analysis
│
|
│
├── src/                                      # (Optional) Utility scripts if refactored from notebook
│   └── ...
│
├── README.md                                 # Project overview (this file)
└── requirements.txt                          # Python dependencies
