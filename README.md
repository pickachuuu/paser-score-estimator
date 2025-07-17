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

👉 **Mention that** these weights were **interpreted** from the manual's textual descriptions and visual examples — not provided numerically.

---

### ⚠️ Methodological Transparency
- PASER estimates are **used for routing, not official pavement audits**.
- **Manhole covers (class 6)** are **ignored** in scoring.
- **Polygon detections are assumed to reflect severity**, with area as a proxy for extent.
- **No subsurface or temporal data** is considered — only surface-level visual cues.
- Heuristic thresholds for PASER mapping are currently:
  ```python
  if avg_score > 0.2: PASER 1
  elif > 0.1: PASER 3
  elif > 0.05: PASER 5
  elif > 0.01: PASER 7
  else: PASER 9
