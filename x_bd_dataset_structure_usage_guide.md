# xBD Dataset – Structure, Annotations, and Modeling Guide

This document explains the **on-disk structure**, **data semantics**, and **intended machine learning usage** of the xBD dataset as organized in this project. It is written to be both **prompt-ready** (for AI systems) and **engineering-ready** (for developers and researchers).

---

## 1. High-Level Dataset Layout

The dataset is organized under a root `xbd/` directory and split by dataset usage and annotation quality tiers.

```text
xbd/
├── train/
├── test/
├── hold/
├── tier1/
│   ├── images/
│   ├── labels/
│   └── masks/
└── tier3/
```

### Directory Purpose

- **train / test / hold**  
  Standard dataset splits for training, validation, and final evaluation.

- **tier1**  
  Highest-quality annotations. Recommended for training and benchmarking.

- **tier3**  
  Lower-confidence annotations. Useful for pretraining or robustness experiments.

---

## 2. Images Directory (`images/`)

Each scene is represented by a *pair* of satellite images captured over the same geographic tile.

```text
images/
├── <scene_id>_pre_disaster.png
└── <scene_id>_post_disaster.png
```

### Image Characteristics

- RGB satellite imagery
- Typically **1024 × 1024 pixels**
- Pixel-aligned between pre- and post-disaster images
- Same filename stem (`scene_id`) for pairing
- Same sensor and spatial resolution

### Semantic Meaning

- **Pre-disaster image**: Baseline condition of the built environment
- **Post-disaster image**: Observed state after the disaster event

These two images together form the **primary visual input** for any damage prediction model.

---

## 3. Labels Directory (`labels/`)

For each image, there is a corresponding JSON annotation file. Filenames mirror the image names.

```text
labels/
├── <scene_id>_pre_disaster.json
└── <scene_id>_post_disaster.json
```

### 3.1 Pre-Disaster JSON Annotations

**Purpose:** Building localization only

Contains:
- Building footprint polygons
- No damage classification

Key fields:
- `features.xy`: building polygons in **image pixel coordinates**
- `features.lng_lat`: building polygons in **geographic coordinates**
- `properties.feature_type = "building"`
- `uid`: unique building identifier

This file answers:

> **“Where are the buildings?”**

---

### 3.2 Post-Disaster JSON Annotations

**Purpose:** Building damage assessment

Contains:
- Same building footprints (matched via `uid`)
- Damage classification per building

Additional field:
- `properties.subtype`: damage label

Typical damage classes:
- `no-damage`
- `minor-damage`
- `major-damage`
- `destroyed`

This file answers:

> **“What happened to each building?”**

---

### Critical Annotation Design Principle

- **Building geometry comes from the pre-disaster annotations**
- **Damage labels come from the post-disaster annotations**
- The `uid` field links the same building across time

This enforces *change-based reasoning* rather than single-image classification.

---

## 4. Masks Directory (`masks/`)

Each pre/post image pair has three raster masks:

```text
masks/
├── <scene_id>_pre_disaster.png
├── <scene_id>_post_disaster.png
└── <scene_id>_post_disaster_rgb.png
```

### 4.1 Pre-Disaster Mask

- White pixels over building footprints
- Black background
- Binary building presence mask

**Use cases:**
- Building segmentation pretraining
- Auxiliary supervision

---

### 4.2 Post-Disaster Mask (Binary)

- Typically all black
- Contains no useful signal

**Recommendation:** Safely ignore.

---

### 4.3 Post-Disaster RGB Mask

- Color-coded building masks
- Each color corresponds to a damage class
- Pixel-level damage supervision

Example semantic mapping:

- Green → No damage
- Yellow → Minor damage
- Orange → Major damage
- Red → Destroyed

This is the **primary ground-truth target** for segmentation-based models.

---

## 5. Intended Model Inputs

### Mandatory Inputs

- Pre-disaster image (RGB)
- Post-disaster image (RGB)

### Optional but Valuable Inputs

- Pre-disaster building mask
- Rasterized building polygons

---

## 6. Intended Model Outputs

The dataset supports multiple modeling strategies.

### Option 1: Building-Level Damage Classification

**Output:**
- One damage label per building polygon

**Training Target:**
- `subtype` field from post-disaster JSON

---

### Option 2: Pixel-Wise Damage Segmentation

**Output:**
- Multi-class segmentation map
- Each building pixel assigned a damage class

**Training Target:**
- Post-disaster RGB mask

---

### Option 3: Hybrid (Production-Oriented)

**Output:**
- Pixel-wise damage segmentation
- Aggregated building-level damage scores

This approach aligns best with real-world disaster response systems.

---

## 7. Conceptual Summary

- **Pre-disaster data defines *where* buildings exist**
- **Post-disaster data defines *what changed***
- The dataset is fundamentally a **change detection + conditional classification** problem

Respecting this design is critical for meaningful model performance.

---

## 8. Prompt-Ready Summary

> This dataset consists of paired high-resolution satellite images captured before and after a disaster event. Pre-disaster annotations define building footprints, while post-disaster annotations assign damage severity labels to the same buildings using shared unique identifiers. Raster masks provide pixel-level supervision for both building presence and damage severity. The primary model input is the aligned pre- and post-disaster image pair, and the expected output is either building-level damage classification or pixel-wise damage segmentation.

