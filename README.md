
# 🧠 Internal Project Notes & Reminders: PASER-Based Cyclist Routing System

This is a reminder/documentation file to track **key decisions**, **assumptions**, and **things to explicitly mention in the final research output**.

---

## ✅ Things to Include in the Research Paper / Report

### 🛣️ PASER Score Derivation
- PASER scores are **not directly annotated by certified engineers**.
- Instead, they are **approximated heuristically** using:
  - Distress type detections from YOLOv5
  - Geometric features from polygon area and shape
  - Severity weights **inferred from the Asphalt PASER Manual** (not official values)
- These weights and thresholds are **subjective**, based on manual interpretation of the PASER manual’s descriptive guidelines and photo examples.

👉 **Clearly state** that no official PASER rating system was encoded — this is a **proxy estimation** aligned with cyclist comfort and routing needs.

---

### ⚠️ Methodological Transparency

- **PASER scores are used solely for route optimization**, not for pavement certification or public infrastructure decisions.
- **Each score applies to a single image (~20–30 meters of road)**, not an entire segment, which is a **methodological departure** from standard PASER applications.
- **Manhole covers (class 6)**, although present in the dataset, are **disregarded in PASER score calculations** because:
  - The PASER manual does not factor manhole presence into surface distress scoring.
  - Their inclusion could skew severity assessments unfairly in areas with dense utility infrastructure.
- **Polygon shape and area are treated as proxies for severity**, with larger and more numerous polygons interpreted as more severe surface degradation.
- The scoring logic does not use subsurface, temporal, or structural condition — it relies purely on visual surface cues.
- Heuristic thresholds used to map the normalized distress score to PASER categories:

---

### 🧭 Project Framing: Image-Level PASER Estimation

> “While PASER scores are typically assigned per road segment (e.g., 100–500 meters) based on overall surface condition, our system estimates a PASER-like score per image (≈20–30m stretch) to support fine-grained analysis and cyclist route optimization. These estimates are generated from detected surface distresses and interpreted heuristically based on the PASER manual. We acknowledge this adaptation as a departure from the traditional PASER protocol.”

---

### 📌 Additional Notes to Include
- If possible, mention that the scoring script allows you to label **~8,000 images** quickly — enabling machine learning even without costly expert annotation.
- If a regression model is used, clarify that it is **trained on these heuristic labels**, not expert ones.
- Future improvements may include:
  - Segment-level aggregation of image-based scores
  - Occlusion handling (e.g., vehicles obstructing view)
  - Incorporating contextual features (e.g., elevation, road type)

---

## 🧾 Summary of Core Assumptions

| Assumption | Explanation |
|------------|-------------|
| Image ≈ 20–30m stretch | Based on field of view and visual depth of standard Street View images |
| Manholes ignored | Not factored into PASER manual scoring |
| Distress area = severity proxy | Larger/more polygons imply more serious degradation |
| One PASER per image | Allows local granularity for route scoring |
| ML goal | Route optimization, not pavement audit |

---

### 📈 What Increases Your Chance of Acceptance
- Call your PASER score a “proxy PASER” or “PASER-like estimate” — not "PASER score" as if it's official
- Clearly state the paper is about cycling route optimization with ML, not pavement rating per se
- Include a Limitations section that is as honest as this conversation
- Optionally, evaluate your method against a few expert-labeled examples if you get them — even just 10–20 helps



# 📄 PASER-Based Cyclist Routing System: Paper Checklist (README)

This is a high-level checklist of important aspects, assumptions, limitations, and implementation details that must be mentioned in the final thesis/paper to maintain transparency and scientific rigor.

---

## ✅ Methodology Summary

- We use **Google Street View images** as the visual input for our system.
- Surface distresses are detected using a **YOLOv5 object detection model** trained on the **SVRD dataset** (from Roboflow).
- Each image is assigned a **heuristic PASER-like score** derived from the detection results, not from official engineering-grade annotations.
- These PASER scores are then used as training targets in a **regression model**.

---

## 🧠 Heuristic PASER Scoring

- **PASER labels are not from civil engineers.**
- We use a **rule-based scoring algorithm** inspired by the **Asphalt PASER Manual**.
- Weights and severity levels are inferred from visual charts and descriptions in the manual — not from any numerical dataset.
- **Score mapping logic (example):**
  ```python
  if avg_score > 0.2: PASER 1
  elif > 0.1: PASER 3
  elif > 0.05: PASER 5
  elif > 0.01: PASER 7
  else: PASER 9
  ```
- This makes our PASER values **heuristic proxies**, not official PASER ratings.

---

## 📸 Image-Level Scoring vs Segment-Based

- PASER is conventionally applied to **entire road segments (100–500m)**.
- Our implementation assigns PASER-like scores **per image (~20–30m)**.
- Justification: This provides **finer granularity** for cyclist-specific routing, but should be **clearly acknowledged as a methodological deviation**.

---

## 🛑 Ignored Classes

- **Manhole covers** (Class ID 5) are detected but **excluded from PASER computation** in line with the PASER manual.

---

## 🗺️ Use in Routing

- Final goal is not pavement assessment but **route optimization for cyclists**.
- PASER scores are **one of multiple inputs**, alongside:
  - Total distance
  - Elevation gain
  - (possibly traffic, if added later)

---

## 🔍 Validation & Limitations

- **No engineer-labeled PASER dataset** was available.
- We are exploring a possible comparison between:
  - Manually engineered PASER labels (limited set)
  - Automatically generated PASER scores (heuristic script)
- If done, include **MAE or correlation** analysis between them.

---

## 🧩 Future Work (Optional to Mention)

- Improve PASER scoring via:
  - Expert annotations
  - Incorporating surface/subsurface info
  - Using video-based road surveys instead of still images

---

## 💡 Summary Reminder for Paper Writing

Be sure to include these phrases or ideas:
- "Heuristic approximation of PASER score"
- "Image-level scoring adapted for fine-grained route optimization"
- "Inspired by, but not equivalent to, official PASER methodology"
- "Rule-based mapping informed by PASER manual"
- "Limitations include absence of expert-labeled PASER ground truth"

📊 Random Forest Results:
MAE:  0.0031
RMSE: 0.0317
R²:   0.9985

📊 Gradient Boosting Results:
MAE:  0.0021
RMSE: 0.0331
R²:   0.9984