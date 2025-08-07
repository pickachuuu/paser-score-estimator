import pandas as pd
import numpy as np
from collections import defaultdict

# Class mapping (you can adjust this)
CLASS_MAP = {
    0: 'longitudinal_crack',
    1: 'transverse_crack',
    2: 'alligator_crack',
    3: 'block_crack',
    4: 'pothole',
    5: 'patch'
}

def compute_area(row):
    return (row['x2'] - row['x1']) * (row['y2'] - row['y1'])

def compute_aspect_ratio(row):
    width = row['x2'] - row['x1']
    height = row['y2'] - row['y1']
    return width / height if height != 0 else 0

def heuristic_score(detections):
    score = 10
    counts = defaultdict(int)
    total_area = 0

    for _, row in detections.iterrows():
        class_name = CLASS_MAP.get(row['class_id'], 'unknown')
        area = compute_area(row)
        total_area += area
        counts[class_name] += 1

    has_alligator = counts['alligator_crack'] > 0
    has_pothole = counts['pothole'] > 0
    has_patch = counts['patch'] > 0
    has_long_crack = counts['longitudinal_crack'] > 1
    has_trans_crack = counts['transverse_crack'] > 1
    has_block = counts['block_crack'] > 0

    if has_alligator and has_pothole:
        score = 1
    elif has_alligator:
        score = 2
    elif has_pothole:
        score = 3
    elif has_patch and (has_long_crack or has_trans_crack):
        score = 4
    elif has_patch:
        score = 5
    elif has_long_crack or has_trans_crack or has_block:
        score = 6
    elif total_area > 0:
        score = 7
    else:
        score = 9 + np.random.uniform(0, 1)

    noise = np.random.normal(0, 0.3)
    score = max(1, min(10, score + noise))
    return round(score, 2)

def generate_training_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Prepare output rows
    output_rows = []

    for image_id, group in df.groupby('image_id'):
        score = heuristic_score(group)

        for _, row in group.iterrows():
            width = row['x2'] - row['x1']
            height = row['y2'] - row['y1']
            area = width * height
            aspect_ratio = width / height if height != 0 else 0

            output_rows.append({
                'image_id': row['image_id'],
                'class_id': row['class_id'],
                'confidence': row['confidence'],
                'x1': row['x1'],
                'y1': row['y1'],
                'x2': row['x2'],
                'y2': row['y2'],
                'width': width,
                'height': height,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'heuristic_score': score
            })

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(output_csv, index=False)
    print(f"Training data CSV generated: {output_csv}")

# Example usage:
generate_training_csv("yolo_detections.csv", "training_data.csv")
