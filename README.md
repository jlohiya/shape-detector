# Shape Detector

A browser-based geometric shape detection engine built in TypeScript — no external CV libraries, no ML models. Pure computer vision from scratch using only browser-native APIs and basic math.

## Demo

Upload any image (or use the built-in test set) and the detector will identify every geometric shape, draw bounding boxes, and report confidence scores in real time.

Detects: **circle · triangle · rectangle · pentagon · star**

---

## How It Works

The `detectShapes()` pipeline runs four stages on every image:

### 1. Binarisation
Each pixel is converted to greyscale using the standard luminance formula (`0.299R + 0.587G + 0.114B`) and thresholded at 128. Pixels below the threshold are "dark" and belong to a shape; everything else is background.

### 2. Connected-Component Labelling
A BFS flood-fill with 8-connectivity groups touching dark pixels into isolated blobs. Each blob is one shape candidate. Components smaller than 150 pixels are discarded as noise.

### 3. Contour Extraction
For each component, a scanline pass collects the topmost/bottommost pixel per column and leftmost/rightmost pixel per row. The resulting boundary points are sorted by polar angle around the centroid to form an ordered polygon.

### 4. Feature Extraction & Classification
Four normalised features are computed per shape:

| Feature | Description | Key signal for |
|---|---|---|
| `circularity` | `4π·area / perimeter²` — 1.0 for a perfect circle | circle vs everything |
| `bbFillRatio` | `fillPixels / bboxArea` — how much of the bounding box is filled | rect (≈1.0), triangle (≈0.50), rotated rect (≈0.62) |
| `solidityRatio` | `fillPixels / convexHullArea` — measures concavity | star (≈0.50) vs all convex shapes (≈1.0) |
| `aspectRatio` | `bboxHeight / bboxWidth` | guards circle (must be ~square) |

A six-rule cascade maps these features to a shape class, ordered from most- to least-unambiguous signal:

```
1. Star        → solidityRatio < 0.72   (only shape with deep concavities)
2. Circle      → circularity > 0.82  +  bbFillRatio ≈ π/4
3. Rectangle   → bbFillRatio > 0.88     (fills bbox almost completely)
4. Triangle    → bbFillRatio < 0.60     (lowest fill of all convex shapes)
5. Rot. rect   → bbFillRatio 0.55–0.88  + very high solidity
6. Pentagon    → moderate circularity   + moderate bbFillRatio
```

---

## Getting Started

```bash
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

### Using the Interface
- **Click** any test image to run detection on it immediately
- **Right-click** test images to select/deselect them for batch evaluation
- **Select All / Deselect All** to manage the batch selection
- **Run Selected Evaluation** to score your selection against ground truth
- **Upload Image** (first tile) to test your own images

---

## Project Structure

```
shape-detector/
├── src/
│   ├── main.ts               # Shape detection algorithm + app bootstrap
│   ├── evaluation.ts         # Scoring logic (F1, IoU, center accuracy)
│   ├── evaluation-manager.ts # Wires evaluation UI to scoring logic
│   ├── evaluation-utils.ts   # IoU, distance, and metric helpers
│   ├── ui-utils.ts           # Selection manager + modal manager
│   ├── test-images-data.ts   # Embedded test image data URLs
│   └── style.css             # UI styles
├── public/
│   └── ground_truth.json     # Expected shapes per test image
└── index.html
```

---

## Evaluation Metrics

Each test image is scored across five dimensions:

| Metric | Weight | Target |
|---|---|---|
| Shape Detection Accuracy (F1) | 40% | F1 ≥ 0.9 |
| Localisation (IoU) | 25% | IoU ≥ 0.8 |
| Center Point Accuracy | 15% | ≤ 5px error |
| Area Calculation | 10% | ≥ 90% accuracy |
| Processing Time | 10% | ≤ 500ms |

---

## Constraints

- No external computer vision libraries (OpenCV, etc.)
- No pre-trained machine learning models
- Browser-native APIs and basic math only
- Works directly with the `ImageData` object format from the Canvas API

---

## Tech Stack

- **TypeScript** — typed throughout
- **Vite** — dev server and bundler
- **Canvas API** — sole image processing primitive