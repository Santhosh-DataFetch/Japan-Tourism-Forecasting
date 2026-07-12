# 🇯🇵 Japan Tourism Forecasting & Analytics

An end-to-end Data Analytics and Machine Learning project that analyzes over two decades of Japan tourism data to uncover visitor trends, evaluate the impact of COVID-19, identify seasonal travel patterns, and forecast future visitor arrivals using Machine Learning.

---

## 📌 Project Overview

Japan's tourism industry has experienced significant fluctuations over the past two decades due to seasonality, global events, and changing travel behavior. This project analyzes historical visitor data published by the Japan National Tourism Organization (JNTO) to generate business insights and build predictive models for future tourism demand.

The project combines Python for data analysis and machine learning with Power BI for interactive dashboard development.

---

## 🎯 Objectives

- Analyze historical tourism trends (2003–2024)
- Identify top tourist source countries
- Measure the impact of COVID-19 on tourism
- Explore seasonal travel patterns
- Build forecasting models for monthly visitor arrivals
- Compare Machine Learning models
- Develop an interactive Power BI dashboard for business decision-making

---

## 📂 Dataset

**Source:** Japan National Tourism Organization (JNTO)

Datasets used:

- Monthly Foreign Visitors by Nationality
- Tourism Purpose Dataset

Data Period:

- **Historical Data:** 2003–2024

---

## 🛠 Tools & Technologies

### Programming

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost

### Data Visualization

- Power BI

### Machine Learning

- Random Forest Regressor
- XGBoost Regressor

---

# 📊 Dashboard Pages

## Page 1 — Executive Overview

- Total Visitors
- Top Source Country
- Peak Tourism Season
- Overall Tourism Trends
- Executive KPIs

---

## Page 2 — Tourism Analysis

- Monthly Visitor Trends
- Country-wise Analysis
- COVID-19 Impact
- Seasonal Patterns
- Geographic Insights

---

## Page 3 — Machine Learning Analysis

- Actual vs Predicted Comparison
- Feature Importance
- Model Evaluation
- Model Performance Comparison

---

## Page 4 — Forecast & Recommendations

- Forecast Overview
- Business Recommendations
- Future Tourism Insights
- Model Selection Summary

---

# 🤖 Machine Learning

## Feature Engineering

Created features including:

- Lag 1
- Lag 3
- Lag 12
- Rolling Mean (3 Months)
- Rolling Mean (12 Months)
- Seasonal Features
- Cyclical Month Encoding
- Trend Variables

---

## Models Used

### Random Forest Regressor

Performance

- MAE: **264,179.61**
- RMSE: **319,959.44**
- R² Score: **0.9366**

---

### XGBoost Regressor

Performance

- MAE: **501,363.28**
- RMSE: **673,714.65**
- R² Score: **0.7189**

---

## Model Selection

Random Forest was selected as the final forecasting model due to its superior predictive performance across all evaluation metrics.

---

# 📈 Key Insights

- Tourism experienced a dramatic decline during the COVID-19 pandemic before recovering strongly.
- Visitor arrivals exhibit clear seasonal patterns throughout the year.
- Recent visitor trends (Lag Features) were the strongest predictors of future tourism demand.
- Random Forest significantly outperformed XGBoost for this dataset.

---

# 💼 Business Recommendations

- Increase tourism marketing before peak travel seasons.
- Improve transportation and accommodation capacity during forecasted demand peaks.
- Continuously retrain forecasting models as new tourism data becomes available.
- Use forecasting insights for resource planning and policy decisions.

---

# 📁 Repository Structure

```
Japan-Tourism-Forecasting
│
├── Data
│   ├── Raw
│   └── Cleaned
│
├── Notebook
│   ├── tourism_analysis.ipynb
│   └── tourism_forecasting.py
│
├── Dashboard
│   ├── Tourism Dashboard.pbix
│   └── Dashboard.pdf
│
├── Images
│   ├── dashboard.png
│   ├── feature_importance.png
│   └── actual_vs_predicted.png
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

# 🚀 Future Improvements

- Forecast multiple years into the future
- Deploy the forecasting model using Streamlit
- Automate monthly data updates
- Compare additional Machine Learning algorithms
- Integrate weather and economic indicators

---

## 👨‍💻 Author

**Santhosh S**

B.Sc Artificial Intelligence & Data Science

📧 santhosh.25bscaids@kprcas.ac.in

🔗 LinkedIn  
https://www.linkedin.com/in/santhosh-analytics-

🔗 GitHub  
https://github.com/Santhosh-DataFetch

---

## ⭐ If you found this project useful, consider giving it a star.
