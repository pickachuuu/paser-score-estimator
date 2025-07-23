import pandas as pd
import numpy as np
from pathlib import Path

# Mapping class ids to names
LABELS = {
    0: "Alligator crack",
    1: "Longitudinal crack",
    2: "Longitudinal patch",
    3: "Pothole",
    4: "Transverse crack",
    5: "Transverse patch",
    6: "Manhole cover"
}

# Distress severity weights
DISTRESS_SEVERITY_WEIGHTS = {
    0: 5, 1: 3, 2: 2, 3: 5, 4: 3, 5: 2, 6: 0
}

IMG_W, IMG_H = 640, 640  # YOLOv5 default input resizing is 640×640 :contentReference[oaicite:5]{index=5}

def bbox_area(row):
    # Normalized to [0, 1]
    return max(0.0, ((row['x2'] - row['x1']) * (row['y2'] - row['y1'])) / (IMG_W * IMG_H))

def heuristic_paser(avg_score):
    if avg_score > 0.15:
        return 1
    elif avg_score > 0.10:
        return 2
    elif avg_score > 0.07:
        return 3
    elif avg_score > 0.05:
        return 4
    elif avg_score > 0.03:
        return 5
    elif avg_score > 0.015:
        return 6
    elif avg_score > 0.007:
        return 7
    elif avg_score > 0.003:
        return 8
    else:
        return 9

def estimate_paser(group):
    if group.empty:
        return 9

    distress_score = 0.0
    distress_count = len(group)

    for _, row in group.iterrows():
        cid = int(row['class_id'])
        weight = DISTRESS_SEVERITY_WEIGHTS.get(cid, 0)
        distress_score += weight * bbox_area(row)

    avg_score = distress_score / distress_count
    return heuristic_paser(avg_score)

def process_csv(csv_path, output_file="heuristic_paser_dataset.csv"):
    df = pd.read_csv(csv_path)
    df['area'] = df.apply(bbox_area, axis=1)

    rows = []
    for image_id, group in df.groupby("image_id"):
        paser = estimate_paser(group)
        row = {"image_id": image_id, "paser_score": paser}
        for cid, label in LABELS.items():
            cls = group[group['class_id'] == cid]
            row[f"{label}_count"] = len(cls)
            row[f"{label}_total_area"] = cls['area'].sum()
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(output_file, index=False)
    print(f"[✓] Saved {output_file}")
    return out

# Run the script
process_csv("yolo_detections.csv")
