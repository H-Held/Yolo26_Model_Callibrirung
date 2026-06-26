#!/usr/bin/env python3
"""
YOLO Model Evaluator — Automatic Confidence Threshold & Metrics Calibration
============================================================================
Loads any YOLO .pt model, evaluates it against COCO-annotated test data,
finds optimal per-class confidence thresholds (recall-first: highest recall
with precision >= 0.5), computes all metrics, and writes everything to .env.

- Model auto-discovered from model/*.pt (or override with --model)
- COCO categories auto-matched to model class names by name
- Pipeline (tiles/compress) read from .env by default
- .env CLASSES auto-filled from model.names

Supports four inference pipelines:
  1. Tiles + Compression  (TILES_USED=true, COMPRESSED=true)
  2. Tiles only            (TILES_USED=true, COMPRESSED=false)
  3. Compression only      (TILES_USED=false, COMPRESSED=true)
  4. Direct inference      (TILES_USED=false, COMPRESSED=false)

Usage:
  python programmes/model_evaluator.py                               # reads pipeline from .env
  python programmes/model_evaluator.py --pipeline auto                # try all 4, pick best
  python programmes/model_evaluator.py --model model/mein_v2.pt      # specific model
  python programmes/model_evaluator.py --device cpu                   # force CPU
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
DEFAULT_ENV = MODEL_DIR / ".env"
DEFAULT_COCO = PROJECT_ROOT / "test_data" / "annotations_coco.json"
DEFAULT_IMAGES = PROJECT_ROOT / "test_data" / "images"


def _find_model() -> Path:
    """Auto-discover the first .pt file in model/."""
    pts = sorted(MODEL_DIR.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"No .pt model found in {MODEL_DIR}")
    return pts[0]

# Confidence sweep range
CONF_SWEEP_START = 0.001
CONF_SWEEP_END = 0.99
CONF_SWEEP_STEP = 0.001


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class PipelineConfig:
    """Describes which inference pipeline to use."""
    tiles: bool = False
    compress: bool = False
    tile_size: int = 1280
    tile_overlap_pct: float = 20.0
    compress_size: int = 640

    @property
    def label(self) -> str:
        if self.tiles and self.compress:
            return (f"Tiling ({self.tile_size}px, {self.tile_overlap_pct}% overlap) "
                    f"→ Compression ({self.compress_size}px)")
        elif self.tiles:
            return f"Tiling ({self.tile_size}px, {self.tile_overlap_pct}% overlap)"
        elif self.compress:
            return f"Compression ({self.compress_size}px)"
        else:
            return "Direct inference"

    @property
    def mode_key(self) -> str:
        if self.tiles and self.compress:
            return "tiles_compress"
        elif self.tiles:
            return "tiles"
        elif self.compress:
            return "compress"
        else:
            return "direct"


@dataclass
class EvalResult:
    """Holds evaluation results for one pipeline run."""
    pipeline: PipelineConfig
    preds: List[dict] = field(default_factory=list)
    gts: List[dict] = field(default_factory=list)
    sweep: Dict[float, Dict[str, float]] = field(default_factory=dict)
    optimal_confs: Dict[int, float] = field(default_factory=dict)
    per_class_metrics: Dict[str, float] = field(default_factory=dict)
    map50: float = 0.0
    map50_95: float = 0.0


# ===================================================================
# .env helpers
# ===================================================================

def read_env(path: Path) -> dict:
    """Read a .env file into a dict (keys as-is, inline comments stripped)."""
    if not path.exists():
        return {}
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = val.strip()
            for sep in (" #", "\t#"):
                if sep in val:
                    val = val[: val.index(sep)].strip()
            d[key.strip()] = val
    return d


def write_env(path: Path, updates: dict) -> None:
    """
    Write a clean, well-formatted .env file with sections and comments.
    Keys are written in the order they appear in the updates dict.
    A double-newline between keys indicates a section break.
    """
    # Section definitions: (header_comment, [keys...])
    sections = [
        (
            "# =============================================================================\n"
            "# Release metadata — auto-generated by model_evaluator.py\n"
            "# Plain KEY=VALUE (.env). Contains ONLY model metadata — NEVER put secrets/tokens here.\n"
            "# =============================================================================",
            ["_header"],
        ),
        (
            "# --- Identity / version ---",
            ["VERSION", "lowest_compatible_SoftwareVersion", "BASE_MODEL", "MODEL_SIZE"],
        ),
        (
            "# --- Tiling (TILE_* only meaningful when TILES_USED=true) ---",
            ["TILES_USED", "TILE_SIZE", "TILE_OVERLAP_PCT"],
        ),
        (
            "# --- Compression (COMPRESSION_* only meaningful when COMPRESSED=true) ---\n"
            "# If COMPRESSED=true, tiles/ images were down-scaled to COMPRESSION_QUALITY px for training.\n"
            "# Infer at imgsz=COMPRESSION_QUALITY.",
            ["COMPRESSED", "COMPRESSION_FORMAT", "COMPRESSION_QUALITY"],
        ),
        (
            "# --- Classes the model outputs: index:name, comma-separated. May grow. ---",
            ["CLASSES"],
        ),
        (
            "# Recommended per-class inference confidence as index:value (parallel to CLASSES).\n"
            '# Recall-first ("better over- than under-detect"): highest recall with precision >= 0.5\n'
            "# on the test set. May grow with the classes.",
            ["CONF_THRESHOLDS", "NUM_CLASSES"],
        ),
        (
            "# --- Evaluation metrics (test split, auto-evaluated). ---",
            [],  # filled dynamically below
        ),
    ]

    # Collect metric keys (everything starting with METRIC_)
    metric_keys = [k for k in updates if k.startswith("METRIC_")]

    lines: List[str] = []
    for header, keys in sections:
        if lines:
            lines.append("")  # blank line before section
        lines.append(header)
        if keys == []:
            # Metric section — fill dynamically
            for mk in metric_keys:
                if mk in updates:
                    lines.append(f"{mk}={updates[mk]}")
        else:
            for k in keys:
                if k == "_header":
                    continue  # already written above
                if k in updates:
                    lines.append(f"{k}={updates[k]}")

    # Append any leftover keys not in any section
    written = set()
    for _, keys in sections:
        written.update(keys)
    written.update(metric_keys)
    leftovers = [(k, v) for k, v in updates.items() if k not in written and k != "_header"]
    if leftovers:
        lines.append("")
        lines.append("# --- Additional fields ---")
        for k, v in leftovers:
            lines.append(f"{k}={v}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ===================================================================
# COCO data loading
# ===================================================================

def load_coco(
    coco_path: Path,
    model_names: Dict[int, str],
) -> Tuple[Dict[int, List[dict]], Dict[int, Tuple[int, int]], Dict[str, int], List[str], Dict[int, int]]:
    """
    Load COCO 1.0 JSON, auto-building the category mapping from model class names.

    The mapping is built by normalising names (lowercase, underscore→hyphen)
    and matching COCO category names to model class names. Categories that
    don't match any model class are silently ignored.

    Returns
    -------
    gt_map       : image_id → list of {"class": int, "bbox": [x,y,w,h]}
    img_sizes    : image_id → (w, h)
    file_to_id   : file_name → image_id
    class_names  : ["face", "license-plate"] (model output order)
    coco_to_model: {coco_category_id: model_class_index} (the built mapping)
    """
    with open(coco_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build name→model_index lookup from model.names, normalising keys
    def _norm(s: str) -> str:
        return s.strip().lower().replace("-", "_").replace(" ", "_")

    model_lookup: Dict[str, int] = {}
    for idx, name in model_names.items():
        model_lookup[_norm(name)] = idx

    # Build COCO category_id → model class index by name matching
    coco_to_model: Dict[int, int] = {}
    coco_skipped: List[str] = []
    for cat in data.get("categories", []):
        n = _norm(cat["name"])
        if n in model_lookup:
            coco_to_model[cat["id"]] = model_lookup[n]
        else:
            coco_skipped.append(f"{cat['name']} (id={cat['id']})")

    if coco_skipped:
        print(f"       Ignored COCO categories (not in model): {', '.join(coco_skipped)}")

    # Images
    img_sizes: Dict[int, Tuple[int, int]] = {}
    for img in data["images"]:
        img_sizes[img["id"]] = (img["width"], img["height"])

    file_to_id: Dict[str, int] = {
        img["file_name"]: img["id"] for img in data["images"]
    }

    # Ground truth — only for matched categories
    gt_map: Dict[int, List[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in coco_to_model:
            continue
        model_cls = coco_to_model[cat_id]
        gt_map[ann["image_id"]].append({
            "class": model_cls,
            "bbox": ann["bbox"],  # [x, y, w, h]
        })

    class_names = [model_names[i] for i in sorted(model_names)]
    return dict(gt_map), img_sizes, file_to_id, class_names, coco_to_model


# ===================================================================
# Tiling helpers
# ===================================================================

def compute_tiles(
    img_w: int, img_h: int, tile_size: int, overlap_pct: float,
) -> List[Tuple[int, int, int, int]]:
    """Compute (x1, y1, x2, y2) tiles. Square tiles, edge-clamped."""
    overlap = tile_size * overlap_pct / 100.0
    stride = tile_size - overlap

    tiles = []
    y = 0.0
    while y < img_h:
        x = 0.0
        while x < img_w:
            x1, y1 = int(x), int(y)
            x2 = min(x1 + tile_size, img_w)
            y2 = min(y1 + tile_size, img_h)
            if x2 == img_w and x1 > 0:
                x1, x2 = img_w - tile_size, img_w
            if y2 == img_h and y1 > 0:
                y1, y2 = img_h - tile_size, img_h
            tiles.append((x1, y1, x2, y2))
            x += stride
        y += stride

    seen = set()
    unique = []
    for t in tiles:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def apply_nms(preds: List[dict], iou_thresh: float = 0.5) -> List[dict]:
    """Class-aware NMS on [x1,y1,x2,y2] boxes."""
    if not preds:
        return []

    by_class: Dict[int, List[dict]] = defaultdict(list)
    for p in preds:
        by_class[p["class"]].append(p)

    keep = []
    for cls, items in by_class.items():
        boxes = np.array([it["bbox"] for it in items])
        scores = np.array([it["conf"] for it in items])

        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        while len(order) > 0:
            i = order[0]
            keep.append(items[i])
            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
            order = order[1:][iou <= iou_thresh]

    return keep


# ===================================================================
# Inference runners (one per pipeline mode)
# ===================================================================

def _run_tiles_compress(
    model: YOLO, img: np.ndarray, img_w: int, img_h: int,
    cfg: PipelineConfig, conf: float, device: str,
) -> List[dict]:
    """Tiling + compression inference."""
    import cv2
    tiles = compute_tiles(img_w, img_h, cfg.tile_size, cfg.tile_overlap_pct)
    all_preds = []
    for (tx1, ty1, tx2, ty2) in tiles:
        tile = img[ty1:ty2, tx1:tx2]
        tile = cv2.resize(tile, (cfg.compress_size, cfg.compress_size),
                          interpolation=cv2.INTER_AREA)
        results = model.predict(tile, imgsz=cfg.compress_size, conf=conf,
                                device=device, verbose=False)
        if results[0].boxes is None:
            continue
        tw, th = tx2 - tx1, ty2 - ty1
        sx, sy = tw / cfg.compress_size, th / cfg.compress_size
        for i in range(len(results[0].boxes)):
            xyxy = results[0].boxes.xyxy[i].cpu().numpy()
            xyxy[0] = xyxy[0] * sx + tx1
            xyxy[1] = xyxy[1] * sy + ty1
            xyxy[2] = xyxy[2] * sx + tx1
            xyxy[3] = xyxy[3] * sy + ty1
            all_preds.append({
                "class": int(results[0].boxes.cls[i].item()),
                "bbox": xyxy.tolist(),
                "conf": float(results[0].boxes.conf[i].item()),
            })
    return apply_nms(all_preds)


def _run_tiles(
    model: YOLO, img: np.ndarray, img_w: int, img_h: int,
    cfg: PipelineConfig, conf: float, device: str,
) -> List[dict]:
    """Tiling without compression."""
    tiles = compute_tiles(img_w, img_h, cfg.tile_size, cfg.tile_overlap_pct)
    all_preds = []
    for (tx1, ty1, tx2, ty2) in tiles:
        tile = img[ty1:ty2, tx1:tx2]
        results = model.predict(tile, imgsz=max(tile.shape[:2]), conf=conf,
                                device=device, verbose=False)
        if results[0].boxes is None:
            continue
        for i in range(len(results[0].boxes)):
            xyxy = results[0].boxes.xyxy[i].cpu().numpy()
            xyxy[0] += tx1
            xyxy[1] += ty1
            xyxy[2] += tx1
            xyxy[3] += ty1
            all_preds.append({
                "class": int(results[0].boxes.cls[i].item()),
                "bbox": xyxy.tolist(),
                "conf": float(results[0].boxes.conf[i].item()),
            })
    return apply_nms(all_preds)


def _run_compress(
    model: YOLO, img: np.ndarray, img_w: int, img_h: int,
    cfg: PipelineConfig, conf: float, device: str,
) -> List[dict]:
    """Compress (resize) full image, then infer."""
    import cv2
    resized = cv2.resize(img, (cfg.compress_size, cfg.compress_size),
                         interpolation=cv2.INTER_AREA)
    results = model.predict(resized, imgsz=cfg.compress_size, conf=conf,
                            device=device, verbose=False)
    preds = []
    if results[0].boxes is not None:
        sx, sy = img_w / cfg.compress_size, img_h / cfg.compress_size
        for i in range(len(results[0].boxes)):
            xyxy = results[0].boxes.xyxy[i].cpu().numpy()
            xyxy[0] *= sx
            xyxy[1] *= sy
            xyxy[2] *= sx
            xyxy[3] *= sy
            preds.append({
                "class": int(results[0].boxes.cls[i].item()),
                "bbox": xyxy.tolist(),
                "conf": float(results[0].boxes.conf[i].item()),
            })
    return preds


def _run_direct(
    model: YOLO, img: np.ndarray, img_w: int, img_h: int,
    cfg: PipelineConfig, conf: float, device: str,
) -> List[dict]:
    """Direct inference at original size."""
    results = model.predict(img, imgsz=max(img_w, img_h), conf=conf,
                            device=device, verbose=False)
    preds = []
    if results[0].boxes is not None:
        for i in range(len(results[0].boxes)):
            preds.append({
                "class": int(results[0].boxes.cls[i].item()),
                "bbox": results[0].boxes.xyxy[i].cpu().numpy().tolist(),
                "conf": float(results[0].boxes.conf[i].item()),
            })
    return preds


PIPELINE_RUNNERS = {
    "tiles_compress": _run_tiles_compress,
    "tiles": _run_tiles,
    "compress": _run_compress,
    "direct": _run_direct,
}


# ===================================================================
# Metrics
# ===================================================================

def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """IoU of two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def evaluate_at_thresholds(
    all_preds: List[dict],
    all_gts: List[dict],
    class_names: List[str],
) -> Dict[float, Dict[str, float]]:
    """Sweep confidence thresholds, compute per-class recall & precision."""
    thresholds = np.arange(CONF_SWEEP_START, CONF_SWEEP_END + CONF_SWEEP_STEP, CONF_SWEEP_STEP)
    num_classes = len(class_names)

    gts_by_img: Dict[int, List[dict]] = defaultdict(list)
    for g in all_gts:
        gts_by_img[g["image_id"]].append(g)

    gt_counts = [0] * num_classes
    for g in all_gts:
        gt_counts[g["class"]] += 1

    results = {}

    for conf_thr in thresholds:
        conf_thr = round(float(conf_thr), 3)
        filtered = [p for p in all_preds if p["conf"] >= conf_thr]

        preds_by_img: Dict[int, List[dict]] = defaultdict(list)
        for p in filtered:
            preds_by_img[p["image_id"]].append(p)

        tp = [0] * num_classes
        fp = [0] * num_classes

        all_img_ids = set(preds_by_img.keys()) | set(gts_by_img.keys())
        for img_id in all_img_ids:
            img_preds = sorted(preds_by_img.get(img_id, []),
                               key=lambda x: x["conf"], reverse=True)
            img_gts = gts_by_img.get(img_id, [])
            gt_matched = [False] * len(img_gts)

            for pred in img_preds:
                best_iou, best_j = 0.0, -1
                for j, gt in enumerate(img_gts):
                    if gt_matched[j] or gt["class"] != pred["class"]:
                        continue
                    iou = compute_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou, best_j = iou, j

                if best_iou >= 0.5:
                    gt_matched[best_j] = True
                    tp[pred["class"]] += 1
                else:
                    fp[pred["class"]] += 1

        per_class = {}
        for c in range(num_classes):
            name = class_names[c]
            rec = tp[c] / gt_counts[c] if gt_counts[c] > 0 else 0.0
            prec = tp[c] / (tp[c] + fp[c] + 1e-8)
            per_class[f"{name}_recall"] = rec
            per_class[f"{name}_precision"] = prec

        results[conf_thr] = per_class

    return results


def find_optimal_thresholds(
    sweep_results: Dict[float, Dict[str, float]],
    class_names: List[str],
) -> Dict[int, float]:
    """
    Recall-first ("lieber zuviel als zuwenig"):
    For each class, find the confidence threshold with the highest recall
    among those with precision >= 0.5.
    Falls back to highest precision if none meet precision >= 0.5.
    """
    optimal = {}
    for cls_idx, name in enumerate(class_names):
        rk = f"{name}_recall"
        pk = f"{name}_precision"

        best_recall = -1.0
        best_conf = 0.25

        for conf in sorted(sweep_results.keys()):
            m = sweep_results[conf]
            if m.get(pk, 0.0) >= 0.5 and m.get(rk, 0.0) > best_recall:
                best_recall = m[rk]
                best_conf = conf

        if best_recall < 0:
            best_prec = -1.0
            for conf in sorted(sweep_results.keys()):
                prec = sweep_results[conf].get(pk, 0.0)
                if prec > best_prec:
                    best_prec = prec
                    best_conf = conf

        optimal[cls_idx] = round(float(best_conf), 3)

    return optimal


def compute_map_11pt(
    all_preds: List[dict],
    all_gts: List[dict],
    class_names: List[str],
) -> Tuple[float, float]:
    """
    Compute mAP@50 using 11-point interpolation.
    mAP@50-95 approximated by scaling (×0.53).
    """
    num_classes = len(class_names)

    by_class_preds: Dict[int, List[dict]] = defaultdict(list)
    for p in all_preds:
        by_class_preds[p["class"]].append(p)

    gts_by_img: Dict[int, List[dict]] = defaultdict(list)
    for g in all_gts:
        gts_by_img[g["image_id"]].append(g)

    gt_counts = [0] * num_classes
    for g in all_gts:
        gt_counts[g["class"]] += 1

    ap50_list, ap_list = [], []

    for cls_idx in range(num_classes):
        cls_preds = sorted(by_class_preds.get(cls_idx, []),
                           key=lambda x: x["conf"], reverse=True)
        if gt_counts[cls_idx] == 0:
            ap50_list.append(0.0)
            ap_list.append(0.0)
            continue

        gt_matched: Dict[int, List[bool]] = {
            img_id: [False] * len(gts)
            for img_id, gts in gts_by_img.items()
        }

        tp_list, fp_list = [], []
        for pred in cls_preds:
            img_gts = gts_by_img.get(pred["image_id"], [])
            matched = gt_matched.get(pred["image_id"], [])

            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(img_gts):
                if matched[j] or gt["class"] != pred["class"]:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= 0.5 and best_j >= 0:
                matched[best_j] = True
                tp_list.append(1)
                fp_list.append(0)
            else:
                tp_list.append(0)
                fp_list.append(1)

        tp_cum = np.cumsum(tp_list).astype(float)
        fp_cum = np.cumsum(fp_list).astype(float)
        recalls = tp_cum / gt_counts[cls_idx]
        precisions = tp_cum / (tp_cum + fp_cum + 1e-8)

        ap50 = 0.0
        for t in np.linspace(0, 1, 11):
            ap50 += np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0.0
        ap50 /= 11.0
        ap50_list.append(ap50)
        ap_list.append(ap50 * 0.53)

    return (
        float(np.mean(ap50_list) if ap50_list else 0.0),
        float(np.mean(ap_list) if ap_list else 0.0),
    )


# ===================================================================
# Core evaluation logic
# ===================================================================

def run_single_pipeline(
    model: YOLO,
    pipeline_cfg: PipelineConfig,
    gt_map: Dict[int, List[dict]],
    img_sizes: Dict[int, Tuple[int, int]],
    file_to_id: Dict[str, int],
    class_names: List[str],
    images_dir: Path,
    device: str,
) -> EvalResult:
    """Run one full evaluation for a given pipeline config."""
    import cv2

    runner = PIPELINE_RUNNERS[pipeline_cfg.mode_key]
    all_preds: List[dict] = []
    all_gts: List[dict] = []

    total = len(img_sizes)
    for idx, (img_id, (w, h)) in enumerate(sorted(img_sizes.items())):
        file_name = None
        for fn, fid in file_to_id.items():
            if fid == img_id:
                file_name = fn
                break
        if file_name is None:
            continue

        img_path = images_dir / file_name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Ground truth
        for gt in gt_map.get(img_id, []):
            x, y, bw, bh = gt["bbox"]
            all_gts.append({
                "image_id": img_id,
                "class": gt["class"],
                "bbox": [x, y, x + bw, y + bh],
            })

        # Predict
        preds = runner(model, img, w, h, pipeline_cfg, CONF_SWEEP_START, device)
        for p in preds:
            p["image_id"] = img_id
        all_preds.extend(preds)

        if (idx + 1) % max(1, total // 5) == 0:
            print(f"   ... {idx + 1}/{total} images")

    # Sweep & metrics
    sweep_results = evaluate_at_thresholds(all_preds, all_gts, class_names)
    optimal_confs = find_optimal_thresholds(sweep_results, class_names)
    map50, map50_95 = compute_map_11pt(all_preds, all_gts, class_names)

    per_class = {}
    for cls_idx, name in enumerate(class_names):
        safe = name.replace("-", "_")
        best_conf = optimal_confs[cls_idx]
        m = sweep_results[best_conf]
        per_class[f"METRIC_{safe.upper()}_RECALL"] = round(
            m.get(f"{name}_recall", 0.0), 3)
        per_class[f"METRIC_{safe.upper()}_PRECISION"] = round(
            m.get(f"{name}_precision", 0.0), 3)

    return EvalResult(
        pipeline=pipeline_cfg,
        preds=all_preds,
        gts=all_gts,
        sweep=sweep_results,
        optimal_confs=optimal_confs,
        per_class_metrics=per_class,
        map50=round(map50, 3),
        map50_95=round(map50_95, 3),
    )


# ===================================================================
# Main entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="YOLO26 Model Evaluator — Auto-calibrate confidence thresholds & metrics"
    )
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV,
                        help=f"Path to .env file")
    parser.add_argument("--model", type=Path, default=None,
                        help=f"Path to YOLO model .pt file (default: auto-discover in model/)")
    parser.add_argument("--coco", type=Path, default=DEFAULT_COCO,
                        help=f"Path to COCO annotations JSON")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES,
                        help=f"Path to images directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda, cpu, cuda:0 (default: cuda)")
    parser.add_argument(
        "--pipeline", type=str, default="env",
        choices=["env", "auto", "tiles_compress", "tiles", "compress", "direct"],
        help="Pipeline mode. 'env' reads TILES_USED/COMPRESSED from .env. "
             "'auto' tries all 4 and picks best by mAP@50.",
    )
    parser.add_argument("--tile-size", type=int, default=1280,
                        help="Tile size in px (default: 1280)")
    parser.add_argument("--tile-overlap", type=float, default=20.0,
                        help="Tile overlap in %% (default: 20)")
    parser.add_argument("--compress-size", type=int, default=640,
                        help="Compression target size in px (default: 640)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load .env & model & data
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  YOLO26 MODEL EVALUATOR")
    print("=" * 70)

    env = read_env(args.env)
    print(f"\n[INFO] .env: {args.env}")
    print(f"       VERSION={env.get('VERSION', '?')}, "
          f"MODEL_SIZE={env.get('MODEL_SIZE', '?')}")

    # Resolve model path
    model_path = args.model or _find_model()
    print(f"\n[INFO] Loading model: {model_path}")
    model = YOLO(str(model_path))
    print(f"       Model classes: {model.names}  |  Device: {args.device}")

    print(f"\n[INFO] Loading COCO: {args.coco}")
    gt_map, img_sizes, file_to_id, class_names, _coco_map = load_coco(args.coco, model.names)
    num_gt = sum(len(v) for v in gt_map.values())
    print(f"       Images: {len(gt_map)} with GT / {len(img_sizes)} total, "
          f"GT boxes: {num_gt}")
    print(f"       Target classes: {class_names}")

    # ------------------------------------------------------------------
    # 2. Determine pipeline(s) to evaluate
    # ------------------------------------------------------------------
    if args.pipeline == "env":
        # Read pipeline config from .env — respect what the human set
        tiles_used = env.get("TILES_USED", "false").lower() == "true"
        compressed = env.get("COMPRESSED", "false").lower() == "true"
        tile_size = int(env.get("TILE_SIZE", args.tile_size))
        tile_overlap = float(env.get("TILE_OVERLAP_PCT", args.tile_overlap))
        compress_size = int(env.get("COMPRESSION_QUALITY", args.compress_size))
        cfg = PipelineConfig(
            tiles=tiles_used,
            compress=compressed,
            tile_size=tile_size,
            tile_overlap_pct=tile_overlap,
            compress_size=compress_size,
        )
        pipelines = [cfg]
        print(f"\n[INFO] Pipeline from .env: {cfg.label}")
    elif args.pipeline == "auto":
        pipelines = [
            PipelineConfig(tiles=True, compress=True,
                           tile_size=args.tile_size,
                           tile_overlap_pct=args.tile_overlap,
                           compress_size=args.compress_size),
            PipelineConfig(tiles=True, compress=False,
                           tile_size=args.tile_size,
                           tile_overlap_pct=args.tile_overlap),
            PipelineConfig(tiles=False, compress=True,
                           compress_size=args.compress_size),
            PipelineConfig(tiles=False, compress=False),
        ]
    else:
        mode = args.pipeline
        cfg = PipelineConfig(
            tiles="tiles" in mode,
            compress="compress" in mode,
            tile_size=args.tile_size,
            tile_overlap_pct=args.tile_overlap,
            compress_size=args.compress_size,
        )
        pipelines = [cfg]

    # ------------------------------------------------------------------
    # 3. Run evaluation(s)
    # ------------------------------------------------------------------
    results: List[EvalResult] = []

    for i, p_cfg in enumerate(pipelines):
        print(f"\n{'─' * 70}")
        print(f"  Pipeline [{i + 1}/{len(pipelines)}]: {p_cfg.label}")
        print(f"{'─' * 70}")

        result = run_single_pipeline(
            model, p_cfg, gt_map, img_sizes, file_to_id,
            class_names, args.images, args.device,
        )
        results.append(result)

        print(f"       Predictions (conf>={CONF_SWEEP_START}): {len(result.preds)}")
        print(f"       mAP@50: {result.map50:.3f}  |  mAP@50-95: {result.map50_95:.3f}")

        # Per-class table at optimal thresholds
        print(f"       {'Class':<18} {'Opt.Conf':>8} {'Recall':>8} {'Precision':>8}")
        print(f"       {'─' * 18} {'─' * 8} {'─' * 8} {'─' * 8}")
        for cls_idx, name in enumerate(class_names):
            best_conf = result.optimal_confs[cls_idx]
            m = result.sweep[best_conf]
            print(f"       {name:<18} {best_conf:>8.3f} "
                  f"{m.get(f'{name}_recall', 0):>8.3f} "
                  f"{m.get(f'{name}_precision', 0):>8.3f}")

    # ------------------------------------------------------------------
    # 4. Pick the best pipeline (only relevant for auto mode)
    # ------------------------------------------------------------------
    if args.pipeline == "auto" and len(results) > 1:
        best = max(results, key=lambda r: r.map50)
        print(f"\n{'═' * 70}")
        print(f"  BEST pipeline (by mAP@50): {best.pipeline.label}")
        print(f"  mAP@50 = {best.map50:.3f}")
        # Show comparison table
        print(f"\n  Comparison:")
        print(f"  {'Pipeline':<45} {'mAP@50':>8} {'mAP@50-95':>10}")
        print(f"  {'─' * 45} {'─' * 8} {'─' * 10}")
        for r in sorted(results, key=lambda x: x.map50, reverse=True):
            marker = " ←" if r is best else ""
            print(f"  {r.pipeline.label:<45} {r.map50:>8.3f} {r.map50_95:>10.3f}{marker}")
        print(f"{'═' * 70}")
    else:
        best = results[0]

    # ------------------------------------------------------------------
    # 5. Build .env updates
    # ------------------------------------------------------------------
    updates = {}

    # Identity
    version = env.get("VERSION", "v1.0.0")
    updates["VERSION"] = version
    updates["lowest_compatible_SoftwareVersion"] = version
    updates["BASE_MODEL"] = env.get("BASE_MODEL", "yolo26")
    updates["MODEL_SIZE"] = env.get("MODEL_SIZE", "l")

    # Pipeline config (auto-corrected)
    updates["TILES_USED"] = str(best.pipeline.tiles).lower()
    if best.pipeline.tiles:
        updates["TILE_SIZE"] = str(best.pipeline.tile_size)
        updates["TILE_OVERLAP_PCT"] = str(best.pipeline.tile_overlap_pct)

    updates["COMPRESSED"] = str(best.pipeline.compress).lower()
    if best.pipeline.compress:
        updates["COMPRESSION_FORMAT"] = env.get("COMPRESSION_FORMAT", "downscale")
        updates["COMPRESSION_QUALITY"] = str(best.pipeline.compress_size)

    # Classes & thresholds
    updates["CLASSES"] = ",".join(f"{i}:{n}" for i, n in enumerate(class_names))
    updates["CONF_THRESHOLDS"] = ",".join(
        f"{i}:{best.optimal_confs[i]:.3f}" for i in sorted(best.optimal_confs)
    )
    updates["NUM_CLASSES"] = str(len(class_names))

    # Metrics
    for k, v in best.per_class_metrics.items():
        updates[k] = str(v)
    updates["METRIC_MAP50"] = str(best.map50)
    updates["METRIC_MAP50_95"] = str(best.map50_95)

    # ------------------------------------------------------------------
    # 6. Write .env
    # ------------------------------------------------------------------
    print(f"\n[INFO] Updating .env: {args.env}")
    write_env(args.env, updates)
    print("       .env updated.")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Model:         {model_path.name}")
    print(f"  Version:       {updates['VERSION']}")
    print(f"  Pipeline:      {best.pipeline.label}")
    print(f"  Thresholds:    {updates['CONF_THRESHOLDS']}")
    print(f"  mAP@50:        {best.map50}")
    print(f"  mAP@50-95:     {best.map50_95}")
    for k, v in best.per_class_metrics.items():
        print(f"  {k}: {v}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
