import joblib
import pandas as pd
import numpy as np

# 1. Load the model
# Make sure the 'paser_regressor.joblib' file is in the same directory
# as your Python script, or provide the full path to the file.
loaded_model = joblib.load("paser_gb_regressor.joblib")

print("✅ Model loaded successfully!")

# Define the exact feature columns based on your training CSV
# These are all columns except 'image_id' and 'paser_score'
feature_columns = [
    "Alligator crack_count", "Alligator crack_total_area",
    "Longitudinal crack_count", "Longitudinal crack_total_area",
    "Longitudinal patch_count", "Longitudinal patch_total_area",
    "Pothole_count", "Pothole_total_area",
    "Transverse crack_count", "Transverse crack_total_area",
    "Transverse patch_count", "Transverse patch_total_area",
    "Manhole cover_count", "Manhole cover_total_area"
]

# 2. Prepare new data for prediction
# This is where you'd put the actual feature values for a new image you want to predict.
# Each inner list represents a single new image/sample.
# The values within each inner list MUST correspond to the 'feature_columns' defined above,
# and in the exact same order.

# --- EXAMPLE 1: Predicting for a single new image ---
print("\n--- Predicting for a single new image ---")
new_image_data_single = {
    "Alligator crack_count": [0],
    "Alligator crack_total_area": [0.0],
    "Longitudinal crack_count": [1],
    "Longitudinal crack_total_area": [0.05],
    "Longitudinal patch_count": [0],
    "Longitudinal patch_total_area": [0.0],
    "Pothole_count": [0],
    "Pothole_total_area": [0.0],
    "Transverse crack_count": [0],
    "Transverse crack_total_area": [0.0],
    "Transverse patch_count": [0],
    "Transverse patch_total_area": [0.0],
    "Manhole cover_count": [0],
    "Manhole cover_total_area": [0.0]
}
# Create a DataFrame for this single new image, ensuring correct column order
new_data_df_single = pd.DataFrame(new_image_data_single, columns=feature_columns)

# Make prediction for the single image
prediction_single = loaded_model.predict(new_data_df_single)
print(f"Predicted paser_score for the single image: {prediction_single[0]:.4f}") # [0] because it returns an array


# --- EXAMPLE 2: Predicting for multiple new images ---
print("\n--- Predicting for multiple new images ---")
# Example data for 2 new images. You would replace these with your actual new data.
# Make sure each inner list corresponds to a single image's features.
new_images_data_multiple = {
    "Alligator crack_count": [0, 1],
    "Alligator crack_total_area": [0.0, 0.15],
    "Longitudinal crack_count": [1, 0],
    "Longitudinal crack_total_area": [0.06, 0.0],
    "Longitudinal patch_count": [0, 0],
    "Longitudinal patch_total_area": [0.0, 0.0],
    "Pothole_count": [0, 0],
    "Pothole_total_area": [0.0, 0.0],
    "Transverse crack_count": [1, 2],
    "Transverse crack_total_area": [0.01, 0.03],
    "Transverse patch_count": [0, 0],
    "Transverse patch_total_area": [0.0, 0.0],
    "Manhole cover_count": [0, 1],
    "Manhole cover_total_area": [0.0, 0.005]
}
# Create a DataFrame for multiple new images, ensuring correct column order
new_data_df_multiple = pd.DataFrame(new_images_data_multiple, columns=feature_columns)

# Make predictions for multiple images
predictions_multiple = loaded_model.predict(new_data_df_multiple)
print("Predicted paser_scores for multiple images:")
for i, score in enumerate(predictions_multiple):
    print(f"  Image {i+1}: {score:.4f}")

# You can then use these predictions in your application.