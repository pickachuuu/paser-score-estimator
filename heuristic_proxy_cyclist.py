import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier  # optional

LABELS = {0:"Alligator_crack",1:"Longitudinal_crack",2:"Longitudinal_patch",
          3:"Pothole",4:"Transverse_crack",5:"Transverse_patch",6:"Manhole_cover"}
DISTRESS_WEIGHTS = {0:7,1:3,2:2,3:8,4:3,5:2,6:0}
IMG_W, IMG_H = 640, 640
IMG_AREA = IMG_W * IMG_H

def bbox_area_px(row):
    return max(0.0, (row['x2']-row['x1'])*(row['y2']-row['y1']))

def compute_features(group):
    group = group.copy()
    group['area_px'] = group.apply(bbox_area_px, axis=1)
    distress_score = sum(
        DISTRESS_WEIGHTS[int(r['class_id'])]*r['area_px'] for _,r in group.iterrows()
    )
    avg_score = distress_score / (IMG_AREA + 1e-6)
    alligator = group.loc[group['class_id']==0, 'area_px'].sum()
    pothole = group.loc[group['class_id']==3, 'area_px'].sum()
    severe_ratio = (alligator + pothole) / (IMG_AREA + 1e-6)
    return {
        'avg_score': avg_score,
        'severe_ratio': severe_ratio,
        'total_distress_px': group['area_px'].sum()
    }

def heuristic_cyclist_score(feats):
    sr = feats['severe_ratio']
    # severity thresholds from cycling comfort research
    if sr > 0.25:
        return (1, 'Unsafe: >25% severe area')
    if sr > 0.18:
        return (2, 'Very poor comfort')
    if sr > 0.10:
        return (3, 'Poor comfort')
    if sr > 0.02:
        return (4, 'Fair comfort')
    avg = feats['avg_score']
    if avg < 0.00005:
        return (9, 'Excellent surface')
    elif avg < 0.0002:
        return (8, '')
    elif avg < 0.0005:
        return (7, '')
    elif avg < 0.001:
        return (6, '')
    elif avg < 0.002:
        return (5, '')
    else:
        return (4, '')

class CyclistProxyEstimator:
    def __init__(self, clf=None):
        self.clf = clf

    def predict(self, feats):
        if self.clf:
            X = np.array([feats[k] for k in self.clf.feature_names_in_]).reshape(1,-1)
            return int(self.clf.predict(X)[0]), 'ML model'
        else:
            return heuristic_cyclist_score(feats)

def process_csv(filepath, clf=None, output_file='cyclist_proxy_scores.csv'):
    df = pd.read_csv(filepath)
    rows = []
    estimator = CyclistProxyEstimator(clf=clf)
    for img_id, grp in df.groupby('image_id'):
        feats = compute_features(grp)
        score, note = estimator.predict(feats)
        row = {'image_id': img_id, 'cyclist_score': score, 'note': note}
        row.update(feats)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_file, index=False)
    print(f"Saved {output_file}")
    return out

def process_csv_for_regression(filepath, clf=None, output_file='cyclist_proxy_regression.csv'):
    df = pd.read_csv(filepath)
    rows = []
    estimator = CyclistProxyEstimator(clf=clf)
    for img_id, grp in df.groupby('image_id'):
        feats = compute_features(grp)
        score, _ = estimator.predict(feats)
        row = {'image_id': img_id, 'cyclist_score': score}
        row.update(feats)
        rows.append(row)
    out = pd.DataFrame(rows)
    # Only keep numeric columns for regression
    out.to_csv(output_file, index=False)
    print(f"Saved regression-ready CSV: {output_file}")
    return out

def train_cyclist_classifier(features_df, labels):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(features_df, labels)
    return clf

# Example usage:
if __name__ == "__main__":
    # Replace 'your_input.csv' with your actual detection CSV
    process_csv_for_regression('yolo_detections.csv')
