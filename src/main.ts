import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

// Public interfaces 

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

// Step 1: Pixel binarisation 
//
// Before detecting shapes we must decide which pixels ARE a shape and which
// are background. We convert each pixel to a single greyscale brightness value
// using the standard luminance formula, then threshold at 128 (mid-grey).
//
// Luminance = R×0.299 + G×0.587 + B×0.114
// (Green is weighted highest because the human eye is most sensitive to it.)
//
// A pixel is "dark" (belongs to a shape) when:
//   • alpha >= 128   → the pixel is at least half-opaque (not transparent bg)
//   • luminance < 128 → the pixel is darker than mid-grey

/**
 * Returns true if the pixel at (x, y) belongs to a shape.
 *
 * @param data - Raw RGBA byte array from ImageData (4 bytes per pixel)
 * @param w    - Image width (used to convert x,y → flat array index)
 * @param x    - Pixel column
 * @param y    - Pixel row
 */
function isDark(
  data: Uint8ClampedArray,
  w: number,
  x: number,
  y: number
): boolean {
  const i = (y * w + x) * 4; // Each pixel occupies 4 bytes: [R, G, B, A]
  if (data[i + 3] < 128) return false; // Transparent → background
  const luminance = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  return luminance < 128;
}

// Step 2: Connected component labelling 
//
// An image may contain several separate shapes. We group dark pixels that
// belong to the SAME shape using a BFS flood-fill:
//
//   • Start at any unvisited dark pixel (the "seed").
//   • Expand to all 8 neighbours (including diagonals) that are also dark.
//   • Collect the whole connected "blob" as one Component.
//   • Repeat until every dark pixel has been visited.
//
// We also track the axis-aligned bounding box (minX/maxX/minY/maxY) of each
// component during BFS at zero extra cost.
//
// Components smaller than `minSize` pixels are noise (specks, anti-aliasing
// artefacts) and are thrown away.

/** A single connected dark region and its bounding box. */
interface Component {
  pixels: Point[]; // Every dark pixel that belongs to this shape
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

/**
 * Finds all connected dark components using BFS flood-fill (8-connectivity).
 *
 * @param data    - Raw RGBA bytes from ImageData
 * @param w       - Image width
 * @param h       - Image height
 * @param minSize - Minimum pixel count; smaller components are discarded
 */
function findComponents(
  data: Uint8ClampedArray,
  w: number,
  h: number,
  minSize = 150
): Component[] {
  // One byte per pixel; set to 1 once that pixel has been enqueued
  const visited = new Uint8Array(w * h);

  // 8-directional neighbour offsets (W, N, E, S + four diagonals)
  const dx = [-1, 0, 1, -1, 1, -1, 0, 1];
  const dy = [-1, -1, -1, 0, 0, 1, 1, 1];

  const components: Component[] = [];

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (visited[idx] || !isDark(data, w, x, y)) continue;

      // BFS from this seed pixel 
      const queue: number[] = [idx];
      visited[idx] = 1;
      const pixels: Point[] = [];
      let minX = x, maxX = x, minY = y, maxY = y;

      for (let head = 0; head < queue.length; head++) {
        const cur = queue[head];
        const cx = cur % w;
        const cy = (cur / w) | 0; // Fast integer division

        pixels.push({ x: cx, y: cy });

        // Grow bounding box to include this pixel
        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;

        // Check all 8 neighbours; enqueue unvisited dark ones
        for (let d = 0; d < 8; d++) {
          const nx = cx + dx[d];
          const ny = cy + dy[d];
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          const ni = ny * w + nx;
          if (!visited[ni] && isDark(data, w, nx, ny)) {
            visited[ni] = 1;
            queue.push(ni);
          }
        }
      }

      if (pixels.length >= minSize) {
        components.push({ pixels, minX, maxX, minY, maxY });
      }
    }
  }

  return components;
}

// Step 3: Contour extraction 
//
// Feature computations (circularity, solidity …) need the BOUNDARY of each
// shape, not every interior pixel. We use a scanline approach:
//
//   For every column x: record the topmost and bottommost dark pixel.
//   For every row    y: record the leftmost and rightmost dark pixel.
//
// This produces a set of boundary points cheaply. They arrive in scan order
// (top→bottom, left→right), which is not useful for polygon maths.
// We therefore sort them by their polar angle around the centroid, turning
// the unordered cloud into a clockwise-ordered polygon.
//
// Sub-sampling: if the boundary has more than 400 points we thin it evenly
// (keep every kth point). This keeps later calculations fast for large shapes
// while preserving the polygon's overall geometry.

/**
 * Returns an ordered boundary polygon for a component.
 *
 * @param comp - The component whose boundary we want
 * @param data - Raw RGBA bytes
 * @param w    - Image width
 * @param cx   - Centroid x (pre-computed by the caller)
 * @param cy   - Centroid y
 */
function extractContour(
  comp: Component,
  data: Uint8ClampedArray,
  w: number,
  cx: number,
  cy: number
): Point[] {
  const { minX, maxX, minY, maxY } = comp;
  const pts: Point[] = [];

  // Topmost and bottommost dark pixel for every column
  for (let x = minX; x <= maxX; x++) {
    for (let y = minY; y <= maxY; y++) {
      if (isDark(data, w, x, y)) { pts.push({ x, y }); break; } // topmost
    }
    for (let y = maxY; y >= minY; y--) {
      if (isDark(data, w, x, y)) { pts.push({ x, y }); break; } // bottommost
    }
  }

  // Leftmost and rightmost dark pixel for every row
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      if (isDark(data, w, x, y)) { pts.push({ x, y }); break; } // leftmost
    }
    for (let x = maxX; x >= minX; x--) {
      if (isDark(data, w, x, y)) { pts.push({ x, y }); break; } // rightmost
    }
  }

  // Sub-sample to at most 400 boundary points (performance guard)
  const step = Math.max(1, Math.floor(pts.length / 400));
  const sampled = pts.filter((_, i) => i % step === 0);

  // Sort by polar angle → ordered clockwise polygon
  return sampled.sort(
    (a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx)
  );
}

// Geometry utilities

/** Euclidean distance between two points. */
function dist(a: Point, b: Point): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

/**
 * Shoelace formula — area of an arbitrary polygon.
 * Used for both the component's fill area estimate and convex hull area.
 */
function polygonArea(pts: Point[]): number {
  let area = 0;
  for (let i = 0; i < pts.length; i++) {
    const j = (i + 1) % pts.length;
    area += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
  }
  return Math.abs(area) / 2;
}

// Step 4a: Convex hull area (for solidityRatio)
//
// The convex hull is the tightest convex polygon that wraps all contour points
// — imagine stretching a rubber band around the shape.
//
// Comparing fill area to hull area gives the "solidity":
//   • Convex shapes (circle, rect, triangle, pentagon): fill ≈ hull → solidity ≈ 1
//   • Star: five deep inward valleys are skipped by the hull → solidity ≈ 0.4–0.65
//
// Algorithm: Graham scan
//   1. Find the lowest-leftmost point as pivot.
//   2. Sort all other points by polar angle around the pivot.
//   3. Walk the sorted list; discard any point that causes a right-turn
//      (cross product ≤ 0) — this keeps only left-turns, forming a convex polygon.

/**
 * Returns the area of the convex hull of a point set.
 * Used to compute solidityRatio = fillPixels / hullArea.
 */
function convexHullArea(pts: Point[]): number {
  if (pts.length < 3) return 0;

  // Step 1: lowest-leftmost point as pivot
  let pivot = pts[0];
  for (const p of pts) {
    if (p.y > pivot.y || (p.y === pivot.y && p.x < pivot.x)) pivot = p;
  }

  // Step 2: sort remaining points by polar angle around pivot
  const sorted = pts
    .filter((p) => p !== pivot)
    .sort(
      (a, b) =>
        Math.atan2(a.y - pivot.y, a.x - pivot.x) -
        Math.atan2(b.y - pivot.y, b.x - pivot.x)
    );

  // Step 3: Graham scan — keep only left-turns
  const hull: Point[] = [pivot, sorted[0]];
  for (let i = 1; i < sorted.length; i++) {
    while (hull.length >= 2) {
      const a = hull[hull.length - 2];
      const b = hull[hull.length - 1];
      const c = sorted[i];
      // Cross product of vectors ab and ac
      const cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
      if (cross <= 0) hull.pop(); // Right-turn or collinear → remove b
      else break;
    }
    hull.push(sorted[i]);
  }

  return polygonArea(hull);
}

// Step 4b: Feature extraction 
//
// We reduce each component to four normalised scalar features.
// All are dimensionless ratios so they are size-independent.
//
// Expected ranges per shape class:
//
// ┌──────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐
// │ Feature      │ circle     │ rectangle  │ triangle   │ pentagon   │ star       │
// ├──────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤
// │ circularity  │ 0.85–0.98  │ 0.60–0.78  │ 0.48–0.68  │ 0.73–0.86  │ 0.25–0.50  │
// │ bbFillRatio  │ 0.76–0.80  │ 0.92–1.00  │ 0.44–0.55  │ 0.72–0.82  │ 0.35–0.52  │
// │ solidityRatio│ 0.96–1.00  │ 0.95–1.00  │ 0.94–1.00  │ 0.95–1.00  │ 0.40–0.68  │
// │ aspectRatio  │ ≈ 1.0      │ any        │ any        │ ≈ 1.0      │ ≈ 1.0      │
// └──────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘
//
// Special case — rotated rectangle:
//   bbFillRatio drops to ~0.55–0.72 because the tilted rect leaves empty
//   triangular corners inside its own axis-aligned bounding box.
//   Circularity and solidity remain rect-like.

interface ShapeFeatures {
  /**
   * Isoperimetric quotient: 4π·fillArea / perimeter²
   * Perfect circle = 1.0; decreases as shape becomes less round.
   */
  circularity: number;

  /**
   * Bounding-box fill: fillPixels / (bboxW × bboxH)
   * How much of the axis-aligned bounding box the shape actually covers.
   * Key split: rect ≈ 1.0 vs circle ≈ 0.78 vs triangle ≈ 0.50.
   */
  bbFillRatio: number;

  /**
   * Solidity: fillPixels / convexHullArea
   * Measures concavity. Only stars score low; all convex shapes score ≈ 1.
   */
  solidityRatio: number;

  /**
   * Aspect ratio: bboxHeight / bboxWidth
   * Guards the circle rule — circles must have a near-square bounding box.
   */
  aspectRatio: number;
}

/**
 * Computes the four shape features for one component.
 *
 * @param comp     - Component (provides bounding box)
 * @param contour  - Ordered boundary polygon from extractContour
 * @param fillArea - Number of dark pixels (component size)
 */
function extractFeatures(
  comp: Component,
  contour: Point[],
  fillArea: number
): ShapeFeatures {
  const bbW = comp.maxX - comp.minX + 1;
  const bbH = comp.maxY - comp.minY + 1;

  // Perimeter = sum of distances between consecutive contour vertices
  let perimeter = 0;
  for (let i = 0; i < contour.length; i++) {
    perimeter += dist(contour[i], contour[(i + 1) % contour.length]);
  }

  // Circularity = 4π·A / P²  (1 for perfect circle, < 1 for all other shapes)
  const circularity =
    perimeter > 0 ? (4 * Math.PI * fillArea) / (perimeter * perimeter) : 0;

  // Solidity — how much of the convex hull is filled (only stars score low)
  const hull = convexHullArea(contour);
  const solidityRatio = hull > 0 ? fillArea / hull : 1;

  // bbFillRatio — fraction of the bounding box covered by dark pixels
  const bbFillRatio = fillArea / (bbW * bbH);

  // Aspect ratio — height / width of the bounding box
  const aspectRatio = bbH / (bbW || 1);

  return { circularity, bbFillRatio, solidityRatio, aspectRatio };
}

// Step 5: Shape classification
//
// Rules are ordered from most-unambiguous signal to least, so each rule only
// needs to handle shapes that escaped all earlier rules.
//
//  Rule │ Shape              │ Primary discriminator
//  ─────┼────────────────────┼──────────────────────────────────────────────
//   1   │ Star               │ solidityRatio < 0.72  (unique: only concave shape)
//   2   │ Circle             │ circularity > 0.82 + bbFillRatio band ≈ π/4
//   3   │ Rectangle (upright)│ bbFillRatio > 0.88    (fills bbox almost fully)
//   4   │ Triangle           │ bbFillRatio < 0.60    (lowest fill of convex shapes)
//   5   │ Rectangle (rotated)│ bbFillRatio 0.55–0.88 + very high solidity
//   6   │ Pentagon           │ moderate circularity + moderate bbFillRatio
//
// Confidence is a linear function of the decisive feature, clamped to avoid
// overconfident predictions on borderline cases.

type ShapeType = "circle" | "triangle" | "rectangle" | "pentagon" | "star";

/**
 * Maps a ShapeFeatures vector to a shape type and confidence score.
 *
 * @param f - Pre-computed features for one component
 */
function classifyShape(f: ShapeFeatures): { type: ShapeType; confidence: number } {

  // Rule 1: Star 
  // A star's inward points create deep concavities → hull is much larger than
  // fill area → solidityRatio is uniquely low (~0.40–0.65).
  // bbFillRatio < 0.60 guards against tiny shapes with inaccurate hull estimates.
  if (f.solidityRatio < 0.72 && f.bbFillRatio < 0.60) {
    const concavity = 1 - f.solidityRatio; // Higher = deeper concavity = more star-like
    return { type: "star", confidence: Math.min(0.96, 0.50 + concavity * 0.65) };
  }

  // Rule 2: Circle
  // A circle has:
  //   • circularity near 1 (smoothest possible boundary)
  //   • bbFillRatio near π/4 ≈ 0.785 (a circle covers ~78.5% of its square bbox)
  //     Upper bound < 0.88 prevents a square (bbFillRatio ≈ 1) from matching.
  //   • Near-square bbox (aspectRatio close to 1) — circle cannot be elongated
  //   • Very high solidity (smooth convex shape, no concavities)
  if (
    f.circularity > 0.82 &&
    f.bbFillRatio > 0.68 && f.bbFillRatio < 0.88 &&
    f.aspectRatio > 0.75 && f.aspectRatio < 1.33 &&
    f.solidityRatio > 0.93
  ) {
    return { type: "circle", confidence: Math.min(0.98, 0.60 + f.circularity * 0.38) };
  }

  //  Rule 3: Axis-aligned rectangle 
  // An upright rectangle fills its own bounding box almost completely
  // (bbFillRatio > 0.88). circularity < 0.82 rules out any near-circle.
  if (f.bbFillRatio > 0.88 && f.circularity < 0.82 && f.solidityRatio > 0.93) {
    return { type: "rectangle", confidence: Math.min(0.96, 0.72 + f.bbFillRatio * 0.22) };
  }

  //  Rule 4: Triangle 
  // A triangle covers roughly HALF its bounding box (bbFillRatio ≈ 0.50).
  // It has low circularity (three sharp corners) but very high solidity (convex,
  // no inward dents). The solidity > 0.88 guard stops stars reaching this rule.
  if (f.bbFillRatio < 0.60 && f.circularity < 0.72 && f.solidityRatio > 0.88) {
    return {
      type: "triangle",
      confidence: Math.min(0.95, 0.60 + (0.72 - f.circularity) * 0.55),
    };
  }

  // Rule 5: Rotated rectangle 
  // When a rectangle is tilted its axis-aligned bbox grows larger than needed,
  // leaving empty triangular corners → bbFillRatio drops to ~0.55–0.72.
  // It still has very high solidity (shape is still a convex rectangle) and
  // low circularity (four sharp right-angle corners).
  // Distinguished from a triangle: higher bbFillRatio (≥ 0.55 vs < 0.60).
  if (
    f.bbFillRatio >= 0.55 && f.bbFillRatio < 0.88 &&
    f.solidityRatio > 0.92 &&
    f.circularity < 0.76
  ) {
    return { type: "rectangle", confidence: Math.min(0.95, 0.68 + f.solidityRatio * 0.25) };
  }

  //  Rule 6: Pentagon
  // Pentagon occupies the gap between circle and rectangle in feature space:
  //   • Moderate circularity (0.72–0.86) — rounder than a rect, less than a circle
  //   • Moderate bbFillRatio (0.68–0.88) — overlaps circle's range but caught here
  //     because the circle rule already required higher circularity
  //   • Very high solidity (convex shape)
  if (
    f.solidityRatio > 0.92 &&
    f.circularity > 0.72 && f.circularity <= 0.86 &&
    f.bbFillRatio > 0.68 && f.bbFillRatio < 0.88
  ) {
    return { type: "pentagon", confidence: Math.min(0.94, 0.64 + f.solidityRatio * 0.28) };
  }

  //  Fallback heuristics
  // Reached only for unusual inputs: heavy noise, partial occlusion, extreme
  // aspect ratios, etc. Each line uses the single most-distinctive feature.
  if (f.solidityRatio < 0.75) return { type: "star",      confidence: 0.50 + (1 - f.solidityRatio) * 0.38 };
  if (f.bbFillRatio  > 0.85)  return { type: "rectangle", confidence: 0.60 };
  if (f.bbFillRatio  < 0.58)  return { type: "triangle",  confidence: 0.55 };
  if (f.circularity  > 0.80)  return { type: "circle",    confidence: 0.60 + f.circularity * 0.18 };
  if (f.circularity  > 0.70)  return { type: "pentagon",  confidence: 0.58 };
  return { type: "rectangle", confidence: 0.50 };
}

// ShapeDetector class (template — only detectShapes body is new)

export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  /**
   * MAIN ALGORITHM TO IMPLEMENT
   * Method for detecting shapes in an image
   * @param imageData - ImageData from canvas
   * @returns Promise<DetectionResult> - Detection results
   *
   * Pipeline:
   *   1. findComponents  — BFS flood-fill to isolate each dark blob
   *   2. extractContour  — scanline boundary tracing + angle-sort
   *   3. extractFeatures — circularity, bbFillRatio, solidityRatio, aspectRatio
   *   4. classifyShape   — 6-rule cascade to label the shape
   */
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();
    const { data, width, height } = imageData;

    const shapes: DetectedShape[] = [];

    // 1. Find connected dark components
    const components = findComponents(data, width, height, 150);

    for (const comp of components) {
      const bbW = comp.maxX - comp.minX + 1;
      const bbH = comp.maxY - comp.minY + 1;

      // Skip components whose bounding box is too small to be a real shape
      if (bbW < 10 || bbH < 10) continue;

      //  2. Compute centroid (average pixel position) 
      const fillArea = comp.pixels.length;
      let cx = 0, cy = 0;
      for (const p of comp.pixels) {
        cx += p.x;
        cy += p.y;
      }
      cx /= fillArea;
      cy /= fillArea;

      // 3. Extract ordered boundary contour 
      const contour = extractContour(comp, data, width, cx, cy);
      if (contour.length < 6) continue; // Too few points for reliable features

      // 4. Extract features and classify
      const features = extractFeatures(comp, contour, fillArea);
      const { type, confidence } = classifyShape(features);

      shapes.push({
        type,
        confidence,
        boundingBox: {
          x: comp.minX,
          y: comp.minY,
          width: bbW,
          height: bbH,
        },
        center: { x: Math.round(cx), y: Math.round(cy) },
        area: fillArea,
      });
    }

    const processingTime = performance.now() - startTime;

    return {
      shapes,
      processingTime,
      imageWidth: width,
      imageHeight: height,
    };
  }

  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}

// Application bootstrap 

class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById(
      "originalCanvas"
    ) as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById(
      "testImages"
    ) as HTMLDivElement;
    this.evaluateButton = document.getElementById(
      "evaluateButton"
    ) as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById(
      "evaluationResults"
    ) as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      this.evaluationResultsDiv
    );

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";

      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);

      this.displayResults(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;

    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${
              shape.type.charAt(0).toUpperCase() + shape.type.slice(1)
            }</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(1)})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html += "<p>No shapes detected.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      let html =
        '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      // Add upload functionality as first grid item
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName
          .replace(/[_-]/g, " ")
          .replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}" 
               onclick="loadTestImage('${imageName}', '${dataUrl}')" 
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const file = new File([blob], name, { type: "image/svg+xml" });

          const imageData = await this.detector.loadImage(file);
          const results = await this.detector.detectShapes(imageData);
          this.displayResults(results);

          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      (window as any).toggleImageSelection = (
        event: MouseEvent,
        imageName: string
      ) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      // Add upload functionality
      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});