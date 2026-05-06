# ⛽ Petrol Consumption Classifier
### End-to-End EDA · KPI Analysis · Decision Tree Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

This project delivers a **complete machine learning pipeline** to classify U.S. state-level petrol consumption as **High** or **Low**, based on socioeconomic and infrastructure features. It combines rigorous exploratory data analysis (EDA), business KPI reporting, and a tuned Decision Tree Classifier — all in a single, reproducible notebook.

> **Use case:** Energy policy analysts and government planners can use this model to identify which states are likely high-consumption targets for fuel tax interventions.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Records | 48 U.S. states |
| Features | 4 numerical |
| Target | `Consumption_Class` — `1` = High, `0` = Low |
| Source | `petrol_consumption.csv` |
| Class Balance | 50% High / 50% Low (median-split, no imbalance) |

**Features used:**

| Feature | Description |
|---|---|
| `Petrol_tax` | State petrol tax rate |
| `Average_income` | Per-capita average income |
| `Paved_Highways` | Miles of paved highway |
| `Population_Driver_licence(%)` | % of population with a driver's licence |

---

## 🗂️ Project Structure

```
petrol-consumption-classifier/
│
├── Petrol_Tree_Classifier.ipynb   # Main notebook (EDA + Model)
├── petrol_consumption.csv         # Raw dataset
├── dt_pipeline.pkl                # Saved best model (pickle)
└── README.md
```

---

## 🔬 Methodology

### 1 — Exploratory Data Analysis
- Distribution plots and correlation heatmap across all numerical features
- No missing values or duplicate rows detected

### 2 — Feature Engineering & KPI Analysis
Custom grouping columns created for segmentation analysis:

| New Column | Bins |
|---|---|
| `tax_group` | Low / Mid / High tax |
| `income_level` | Low / Mid / Upper-Mid / High |
| `highway_level` | Low / Mid / High / Very High |
| `license_level` | Low / Mid / High / Very High |

KPI dashboards built for:
- High vs Low consumption rates per group
- Cross-segment heatmaps (Tax × Income, Highway × Licence)
- Average income comparison between High and Low consumers

### 3 — Machine Learning Pipeline

```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), num_cols)
    ])),
    ('clf', DecisionTreeClassifier(random_state=42))
])
```

### 4 — Hyperparameter Tuning (GridSearchCV)

| Parameter | Search Space |
|---|---|
| `criterion` | `gini`, `entropy` |
| `max_depth` | 1 – 9 |
| `min_samples_split` | 2, 4, 6, 8 |
| `min_samples_leaf` | 1, 2, 3, 4 |

Cross-validation: **5-fold stratified**, scoring = `accuracy`

---

## 📈 Results

| Metric | Score |
|---|---|
| Train Accuracy | See notebook output |
| Test Accuracy | See notebook output |
| ROC AUC | See notebook output |

> Full confusion matrix (counts + normalised), ROC curve, and classification report are rendered inside the notebook.

---

## 💡 Key Insights

1. **Income is the strongest predictor** — higher-income states consistently show higher petrol consumption.
2. **Low petrol tax correlates with higher consumption** — tax policy has measurable impact.
3. **More paved highways → more driving → more fuel** — infrastructure supply drives demand.
4. **High driver licence rates amplify consumption** across all income segments.
5. **Balanced classes by design** — median-split ensures no class imbalance for the classifier.

---

## ✅ Strategic Recommendations

1. Target fuel tax increases in **high-income, high-licence** states for maximum policy impact.
2. Monitor states with **very high highway coverage** — road supply actively drives consumption.
3. Invest in **public transport** in states with high driver licence rates.
4. Deploy the saved `dt_pipeline.pkl` for real-time consumption class prediction.
5. Combine income and highway features for a more precise policy intervention model.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Notebook
```bash
git clone https://github.com/YOUR_USERNAME/petrol-consumption-classifier.git
cd petrol-consumption-classifier
jupyter notebook Petrol_Tree_Classifier.ipynb
```

### Load the Saved Model
```python
import pickle

with open('dt_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict on new data
# X_new = [[tax, income, highways, licence_pct]]
prediction = model.predict(X_new)
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` & `numpy` | Data manipulation |
| `matplotlib` & `seaborn` | Visualization |
| `scikit-learn` | ML pipeline, tuning, evaluation |
| `pickle` | Model serialization |
| `Jupyter Notebook` | Interactive development |

---

## 👤 Author

**[MOHAMED AHMED MAROF ]**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/MOHAMEDMAROF)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/MOHAMEDMAROF)

---

## 📄 License

This project is licensed under the MIT License.
