import os
import glob
import pandas as pd
import numpy as np
from shapely.geometry import Polygon

# Mapping of class IDs to names
LABELS = {
    0: "Alligator crack",
    1: "Longitudinal crack",
    2: "Longitudinal patch",
    3: "Pothole",
    4: "Transverse crack",
    5: "Transverse patch",
    6: "Manhole cover"
}

# Scoring weight (tunable)
DISTRESS_SEVERITY_WEIGHTS = {
    0: 5,  # Alligator crack
    1: 3,  # Longitudinal crack
    2: 2,  # Longitudinal patch
    3: 5,  # Pothole
    4: 3,  # Transverse crack
    5: 2,  # Transverse patch
    6: 0   # Manhole cover (ignored in PASER)
}

def extract_geometry_features(coords):
    try:
        points = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
        poly = Polygon(points)
        area = poly.area
        perimeter = poly.length
        compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
        minx, miny, maxx, maxy = poly.bounds
        width = maxx - minx
        height = maxy - miny
        aspect_ratio = width / height if height != 0 else 0
        return area, aspect_ratio, compactness
    except:
        return 0, 0, 0

def process_label_file(file_path):
    features = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            area, aspect_ratio, compactness = extract_geometry_features(coords)
            features.append({
                'class_id': class_id,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'compactness': compactness
            })
    return features

def estimate_paser(detections):
    if not detections:
        return 10  # Perfect road

    score = 10
    distress_score = 0
    distress_count = 0

    for d in detections:
        class_id = d['class_id']
        weight = DISTRESS_SEVERITY_WEIGHTS.get(class_id, 0)
        distress_score += weight * d['area']
        distress_count += 1

    # Normalize and convert to PASER
    avg_score = distress_score / max(distress_count, 1)

    # Heuristic mapping
    if avg_score > 0.2:
        return 1  # Severe distress
    elif avg_score > 0.1:
        return 3
    elif avg_score > 0.05:
        return 5
    elif avg_score > 0.01:
        return 7
    else:
        return 9

def aggregate_image_features(file_path):
    detections = process_label_file(file_path)
    image_name = os.path.basename(file_path).replace(".txt", "")
    paser_score = estimate_paser(detections)

    # Also return aggregated counts for learning use
    agg = {'image_id': image_name, 'paser_score': paser_score}
    for cid in LABELS:
        relevant = [d for d in detections if d['class_id'] == cid]
        agg[f"{LABELS[cid]}_count"] = len(relevant)
        agg[f"{LABELS[cid]}_total_area"] = sum(d['area'] for d in relevant)
    return agg

def run_batch(input_dir, output_file="heuristic_paser_dataset.csv"):
    results = []
    for path in glob.glob(os.path.join(input_dir, "*.txt")):
        row = aggregate_image_features(path)
        results.append(row)
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Saved PASER dataset: {output_file}")
    return df

# Example usage
# run_batch("train/labels")
