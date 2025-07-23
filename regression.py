import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# Load dataset
df = pd.read_csv("heuristic_paser_dataset.csv")

# Drop non-feature columns
X = df.drop(columns=["image_id", "paser_score"])
y = df["paser_score"]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_val)

# Evaluation
mae = mean_absolute_error(y_val, y_pred)
# Calculate MSE first, then take the square root for RMSE
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse) # Calculate RMSE manually
r2 = r2_score(y_val, y_pred)

print("\n📊 Evaluation Results:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# Optional: Save the model
joblib.dump(model, "paser_regressor.joblib")
print("\n✅ Model saved as paser_regressor.joblib")