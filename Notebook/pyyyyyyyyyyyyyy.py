import pandas as pd #for EDA
import numpy as np #numeric calculations

import matplotlib.pyplot as plt  #visuals
import seaborn as sns

from sklearn.model_selection import train_test_split  #ML model(test and training )
from sklearn.linear_model import LinearRegression  #(1st model)
from sklearn.ensemble import RandomForestRegressor  #2nd model (slightly poweful)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
import joblib






df = pd.read_csv(r"C:\Users\santh\Downloads\Japan Tourism Analysis\Data\Cleaned\JTD_cleaned_1.csv")
print(df.head())
print(df.info())






df.dropna(inplace=True)
print(df.isnull().sum())


df["Month"] = df["Month"].str.strip()



month_map = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "september": 9,
    "October": 10,
    "november": 11,
    "december": 12
}

df["Month_Num"] = df["Month"].map(month_map)
df["Month_Num"] = df["Month_Num"].astype(int)

df["Date"] = pd.to_datetime(
    dict(
        year=df["Year"],
        month=df["Month_Num"],
        day=1
    )
)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print(df.describe())


#Top 10 Countries
top_countries = (
    df.groupby("Country")["Visitors"]
      .sum()
      .sort_values(ascending=False)
)

print(top_countries.head(10))

top10 = top_countries.head(10)

plt.figure(figsize=(10,6))
top10.plot(kind="bar")

plt.title("Top 10 Tourist Source Countries")
plt.ylabel("Visitors")


#Monthly visitors Trend

monthly_visitors = (
    df.groupby("Date")["Visitors"]
      .sum()
)

plt.figure(figsize=(14,6))

plt.plot(
    monthly_visitors.index,
    monthly_visitors.values
)

plt.title("Monthly Visitors to Japan")
plt.xlabel("Date")
plt.ylabel("Visitors")





         #COVID Impact


covid = (
    df.groupby("Year")["Visitors"]
      .sum()
)

plt.figure(figsize=(10,5))

covid.plot(kind="bar")

plt.title("Visitors by Year")
plt.ylabel("Visitors")







##################### ML MODEL ########################



forecast_df = (
    df.groupby("Date")["Visitors"].sum().reset_index()
)

forecast_df["Year"] = forecast_df["Date"].dt.year
forecast_df["Month"] = forecast_df["Date"].dt.month

forecast_df["Month_Start"] = forecast_df["Date"].dt.is_month_start.astype(int)

forecast_df["Year_Trend"] = np.arange(len(forecast_df))

forecast_df["Peak_Season"] = (
    forecast_df["Month"].isin([3,4,7,8,10,11])
).astype(int)

forecast_df["Month_Sin"] = np.sin(
    2 * np.pi * forecast_df["Month"] / 12
)

forecast_df["Month_Cos"] = np.cos(
    2 * np.pi * forecast_df["Month"] / 12
)


forecast_df["Lag1"] = (
    forecast_df["Visitors"].shift(1)
)

forecast_df["Lag3"] = (
    forecast_df["Visitors"].shift(3)
)

forecast_df["Lag12"] = (
    forecast_df["Visitors"].shift(12)
)



forecast_df["Rolling3"] = (
    forecast_df["Visitors"]
    .rolling(3)
    .mean()
)

forecast_df["Rolling12"] = (
    forecast_df["Visitors"]
    .rolling(12)
    .mean()
)








forecast_df.dropna(inplace=True)









X = forecast_df[
[
"Year",
"Month",
"Month_Sin",
"Month_Cos",
"Year_Trend",
"Peak_Season",
"Lag1",
"Lag3",
"Lag12",
"Rolling3",
"Rolling12"
]
]

y = forecast_df["Visitors"]








split_index = int(
    len(forecast_df) * 0.8
)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


#Random Forest

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_predictions = rf.predict(X_test)



rf_mae = mean_absolute_error(y_test, rf_predictions)

rf_rmse = np.sqrt(
    mean_squared_error(y_test, rf_predictions)
)

rf_r2 = r2_score(y_test, rf_predictions)




print("\nRandom Forest Results")
print("----------------------")
print(f"MAE : {rf_mae:,.2f}")
print(f"RMSE: {rf_rmse:,.2f}")
print(f"R²  : {rf_r2:.4f}")


#XGB Boost

xgb = XGBRegressor(

    n_estimators=800,

    learning_rate=0.03,

    max_depth=4,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42
)


xgb.fit(
    X_train,
    y_train
)


importance = pd.Series(
    xgb.feature_importances_,
    index=X.columns
).sort_values()

predictions = xgb.predict(X_test)




plt.figure(figsize=(10,6))
importance.plot(kind="barh")

plt.title("XGBoost Feature Importance")
plt.xlabel("Importance Score")

plt.tight_layout()
#plt.show()



plt.figure(figsize=(14,6))

plt.plot(
    y_test.values,
    label="Actual",
    linewidth=3
)

plt.plot(
    rf_predictions,
    '--',
    linewidth=2,
    label="Random Forest"
)

plt.plot(
    predictions,
    '--',
    linewidth=2,
    label="XGBoost"
)

plt.title("Actual vs Predicted Comparison")

plt.xlabel("Time")

plt.ylabel("Visitors")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()



mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)





results = pd.DataFrame({
    "Date": forecast_df.iloc[split_index:]["Date"],
    "Actual": y_test.values,
    "RandomForest": rf_predictions,
    "XGBoost": predictions
})

results.to_csv("Forecast_Results.csv", index=False)

joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(xgb, "xgboost_model.pkl")

