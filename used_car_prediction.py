import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# LOAD DATASET
# -------------------------------

df = pd.read_csv("used_car_price_dataset.csv")

# Display first 5 rows
print(df.head())

# Check number of rows and columns
print("Dataset Shape:", df.shape)

# Get information about the dataset
print("\nDataset Information:")
print(df.info())

# Get statistical summary
print("\nStatistical Summary:")
print(df.describe())

# -------------------------------
# DATA CLEANING
# -------------------------------

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# -------------------------------
# DATA VISUALIZATION
# -------------------------------

# 1. Year vs Selling Price
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Year",
    y="Selling_Price"
)

plt.title("Year vs Selling Price")
plt.xlabel("Manufacturing Year")
plt.ylabel("Selling Price")
plt.show()


# 2. Kilometers Driven vs Selling Price
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Kms_Driven",
    y="Selling_Price"
)

plt.title("Kilometers Driven vs Selling Price")
plt.xlabel("Kilometers Driven")
plt.ylabel("Selling Price")
plt.show()


# 3. Fuel Type vs Selling Price
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Fuel_Type",
    y="Selling_Price"
)

plt.title("Fuel Type vs Selling Price")
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price")
plt.show()
# -------------------------------
# CORRELATION ANALYSIS
# -------------------------------

# Select numerical columns
numeric_df = df.select_dtypes(include=["number"])

# Calculate correlation
correlation = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation)

# Display correlation using heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()
# -------------------------------
# CONVERT CATEGORICAL DATA
# -------------------------------

# Convert categorical columns into numerical values
df_encoded = pd.get_dummies(
    df,
    columns=["Fuel_Type", "Seller_Type", "Transmission"],
    drop_first=True
)

print("\nEncoded Dataset:")
print(df_encoded.head())

print("\nNew Columns:")
print(df_encoded.columns)
# -------------------------------
# CATEGORICAL DATA ENCODING
# -------------------------------

df_encoded = pd.get_dummies(
    df,
    columns=["Fuel_Type", "Seller_Type", "Transmission"],
    drop_first=True
)

print("\nEncoded Dataset:")
print(df_encoded.head())

print("\nNew Columns:")
print(df_encoded.columns)
# -------------------------------
# FEATURES AND TARGET
# -------------------------------

# Remove Car_Name because it is text and not used for prediction
df_encoded = df_encoded.drop("Car_Name", axis=1)

# X = input features
X = df_encoded.drop("Selling_Price", axis=1)

# y = target variable
y = df_encoded["Selling_Price"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)
# -------------------------------
# TRAIN-TEST SPLIT
# -------------------------------

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
# -------------------------------
# RANDOM FOREST REGRESSION
# -------------------------------

from sklearn.ensemble import RandomForestRegressor

# Create the model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")
# -------------------------------
# MAKE PREDICTIONS
# -------------------------------

y_pred = model.predict(X_test)

print("\nPredicted Prices:")
print(y_pred[:10])
# -------------------------------
# MODEL EVALUATION
# -------------------------------

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)
# -------------------------------
# ACTUAL VS PREDICTED PRICES
# -------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price")

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.show()
# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

feature_importance = feature_importance.sort_values(ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(9, 5))

feature_importance.plot(kind="bar")

plt.title("Feature Importance in Used Car Price Prediction")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# -------------------------------
# PREDICT PRICE FOR A NEW CAR
# -------------------------------

new_car = pd.DataFrame({
    "Year": [2020],
    "Present_Price": [8.50],
    "Kms_Driven": [30000],
    "Owner": [0],
    "Fuel_Type_Diesel": [False],
    "Fuel_Type_Petrol": [True],
    "Seller_Type_Individual": [True],
    "Transmission_Manual": [True]
})

predicted_price = model.predict(new_car)

print("\nNew Car Details:")
print(new_car)

print("\nPredicted Selling Price:", round(predicted_price[0], 2), "lakh")