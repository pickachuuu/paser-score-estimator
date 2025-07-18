
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
