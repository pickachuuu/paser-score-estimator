import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# Define the 14 feature columns (must match your inference script)
feature_columns = [
    "Alligator crack_count", "Alligator crack_total_area",
    "Longitudinal crack_count", "Longitudinal crack_total_area",
    "Longitudinal patch_count", "Longitudinal patch_total_area",
    "Pothole_count", "Pothole_total_area",
    "Transverse crack_count", "Transverse crack_total_area",
    "Transverse patch_count", "Transverse patch_total_area",
    "Manhole cover_count", "Manhole cover_total_area"
]

# Load dataset
df = pd.read_csv("cyclist_proxy_regression.csv")

# Use only the 14 features for X
X = df[feature_columns]
y = df["cyclist_score"]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)

# Train Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_val)

# Evaluation function
def eval_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {name} Results:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

# Print results
eval_model("Random Forest", y_val, rf_pred)
eval_model("Gradient Boosting", y_val, gb_pred)

# Save the Gradient Boosting model (or whichever you prefer)
joblib.dump(gb_model, "paser_gb_regressor.joblib")
print("\n✅ Gradient Boosting model saved as paser_gb_regressor.joblib")