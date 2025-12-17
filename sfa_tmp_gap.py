import os
import glob
import csv
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

# =========================================
# CONFIG
# =========================================

IMG_DIR = Path("/cluster/pixstor/madrias-lab/Hasibur/STARDNet/Datasets/mvaaod/test")

# ground truth labels in YOLO format without confidence
LABEL_GT_DIR = Path("/cluster/pixstor/madrias-lab/Hasibur/STARDNet/Datasets/mvaaod/test")

# baseline predictions
PRED_BASE_DIR = Path("/cluster/pixstor/madrias-lab/Hasibur/STARDNet/motion_model/.stardnet/v12/labels")

# our model predictions
PRED_OURS_DIR = Path("/cluster/pixstor/madrias-lab/Hasibur/STARDNet/motion_model/.stardnet/std/labels")

# root output folder for everything
ROOT_OUT = Path("sfa_stat")

# feature map parameters
HF_DOWNSCALE = 4

CLUTTER_Q_FPN = 0.85       # currently not used for masks, but kept if you want later
CLUTTER_Q_DRFB = 0.80
OBJECT_Q_DRFB_STAB = 0.90

MIN_CONF_WEIGHT = 0.05
CONF_GAMMA = 1.5

DRFB_BLEND_HF = 0.6
DRFB_BLEND_D = 0.7

STAB_POW = 1.4
STAB_CONF_GAMMA = 1.7
STAB_BASE_OUTSIDE = 0.05
GAUSS_SIGMA = 0.8

GT_LINK_IOU_TH = 0.3  # IoU to link ground truth across frames

# evaluation thresholds
CONF_TH_EVAL = 0.25
IOU_TH_EVAL = 0.5

GOOD_F1 = 0.6
BAD_F1 = 0.2
ALPHA = 0.4
MARGIN_SCORE = 0.10


# =========================================
# HELPERS FOR CLIP ID AND FRAME INDEX
# =========================================

def split_clip_frame(img_id: str):
    """
    For names like 'Clip_42_00269' return:
      clip_id  = 'Clip_42'
      frame_idx = 269

    If it cannot parse, returns:
      clip_id  = img_id
      frame_idx = -1
    """
    parts = img_id.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        clip_id = "_".join(parts[:-1])
        frame_idx = int(parts[-1])
    else:
        clip_id = img_id
        frame_idx = -1
    return clip_id, frame_idx


# =========================================
# BASIC IO HELPERS FOR EVAL
# =========================================

def load_yolo_boxes_basic(txt_path: str, has_conf: bool = False):
    """
    For eval: load YOLO boxes in normalized format from txt_path.
    Returns list of dicts with keys cls xc yc w h conf.
    """
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if has_conf and len(parts) > 5 else 1.0
            boxes.append(
                {"cls": cls, "xc": x, "yc": y, "w": w, "h": h, "conf": conf}
            )
    return boxes


def yolo_to_xyxy_dict(box, img_w: int, img_h: int):
    xc = box["xc"] * img_w
    yc = box["yc"] * img_h
    bw = box["w"] * img_w
    bh = box["h"] * img_h
    x1 = xc - bw / 2.0
    y1 = yc - bh / 2.0
    x2 = xc + bw / 2.0
    y2 = yc + bh / 2.0
    return np.array([x1, y1, x2, y2], dtype=float)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def eval_frame(gt_boxes, pred_boxes, img_w: int, img_h: int,
               conf_th: float = 0.25, iou_th: float = 0.5):
    preds = [b for b in pred_boxes if b["conf"] >= conf_th]
    gts = gt_boxes

    gt_used = [False] * len(gts)
    tp = 0
    fp = 0
    fn = 0
    ious_matched = []

    for p in preds:
        best_iou = 0.0
        best_gi = None
        pxy = yolo_to_xyxy_dict(p, img_w, img_h)
        for gi, g in enumerate(gts):
            if gt_used[gi]:
                continue
            if p["cls"] != g["cls"]:
                continue
            gxy = yolo_to_xyxy_dict(g, img_w, img_h)
            iou_val = iou_xyxy(pxy, gxy)
            if iou_val > best_iou:
                best_iou = iou_val
                best_gi = gi
        if best_iou >= iou_th and best_gi is not None:
            tp += 1
            gt_used[best_gi] = True
            ious_matched.append(best_iou)
        else:
            fp += 1

    fn = sum(not u for u in gt_used)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    mean_iou = float(np.mean(ious_matched)) if ious_matched else 0.0

    return dict(tp=tp, fp=fp, fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
                mean_iou=mean_iou,
                gt_matched=gt_used)


# =========================================
# FEATURE MAP HELPERS
# =========================================

def load_yolo_labels_for_fmaps(label_path: Path, img_w: int, img_h: int) -> List[Tuple[float, float, float, float, float]]:
    """
    Load YOLO labels and return (x1, y1, x2, y2, conf) boxes in pixel coordinates.
    Lines are: cls xc yc w h [conf]
    """
    boxes: List[Tuple[float, float, float, float, float]] = []
    if not label_path.is_file():
        return boxes

    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) not in (5, 6):
                continue
            _, xc, yc, w, h = parts[:5]

            xc = float(xc) * img_w
            yc = float(yc) * img_h
            w = float(w) * img_w
            h = float(h) * img_h

            x1 = xc - w / 2.0
            y1 = yc - h / 2.0
            x2 = xc + w / 2.0
            y2 = yc + h / 2.0

            conf = float(parts[5]) if len(parts) == 6 else 1.0
            boxes.append((x1, y1, x2, y2, conf))

    return boxes


def importance_from_patch(bgr_patch: np.ndarray, patch_size: int = 64) -> Optional[np.ndarray]:
    h, w = bgr_patch.shape[:2]
    if h <= 1 or w <= 1:
        return None

    patch = cv2.resize(bgr_patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    gray_f = gray.astype(np.float32) / 255.0

    edges = cv2.Canny(gray, 60, 160).astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(edges, (5, 5), 0)

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap = np.abs(lap)
    if lap.max() > 0:
        lap /= lap.max()
    lap = cv2.GaussianBlur(lap, (5, 5), 0)

    imp = 0.6 * edges + 0.4 * lap

    yy, xx = np.mgrid[0:patch_size, 0:patch_size].astype(np.float32)
    cx = (patch_size - 1) / 2.0
    cy = (patch_size - 1) / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist /= dist.max() + 1e-6
    interior = 1.0 - dist

    imp = 0.7 * imp + 0.3 * (interior * gray_f)

    imp -= imp.min()
    max_val = imp.max()
    if max_val <= 0:
        return None
    imp /= max_val

    q = float(np.quantile(imp, 0.90))
    mask = imp >= q
    sparse = np.zeros_like(imp)
    sparse[mask] = imp[mask]

    kernel = np.ones((3, 3), np.uint8)
    sparse = cv2.dilate(sparse, kernel, iterations=1)
    sparse = cv2.GaussianBlur(sparse, (9, 9), 0)

    if sparse.max() > 0:
        sparse /= sparse.max()

    imp_full = cv2.resize(sparse, (w, h), interpolation=cv2.INTER_LINEAR)
    return imp_full.astype(np.float32)


def compute_global_hf(img: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ds_h = max(8, H // HF_DOWNSCALE)
    ds_w = max(8, W // HF_DOWNSCALE)
    gray_small = cv2.resize(gray, (ds_w, ds_h), interpolation=cv2.INTER_AREA)

    edges = cv2.Canny(gray_small, 60, 160).astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(edges, (5, 5), 0)

    lap = cv2.Laplacian(gray_small, cv2.CV_32F, ksize=3)
    lap = np.abs(lap)
    if lap.max() > 0:
        lap /= lap.max()
    lap = cv2.GaussianBlur(lap, (5, 5), 0)

    hf_small = 0.7 * edges + 0.3 * lap
    hf_small = cv2.GaussianBlur(hf_small, (7, 7), 0)

    hf = cv2.resize(hf_small, (W, H), interpolation=cv2.INTER_LINEAR)
    hf -= hf.min()
    if hf.max() > 0:
        hf /= hf.max()

    return hf


def compute_drfb_response(img: np.ndarray,
                          boxes: List[Tuple[float, float, float, float, float]],
                          hf: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    resp = np.zeros((h, w), dtype=np.float32)

    for (x1, y1, x2, y2, conf) in boxes:
        xx1 = max(0, int(np.floor(x1)))
        yy1 = max(0, int(np.floor(y1)))
        xx2 = min(w, int(np.ceil(x2)))
        yy2 = min(h, int(np.ceil(y2)))
        if xx2 <= xx1 or yy2 <= yy1:
            continue

        crop = img[yy1:yy2, xx1:xx2]
        imp = importance_from_patch(crop)
        if imp is None:
            continue

        conf_w = max(MIN_CONF_WEIGHT, min(float(conf), 1.0))
        imp *= conf_w ** CONF_GAMMA

        region = resp[yy1:yy2, xx1:xx2]
        if region.shape != imp.shape:
            imp = cv2.resize(imp, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_LINEAR)
        np.maximum(region, imp, out=region)
        resp[yy1:yy2, xx1:xx2] = region

    if resp.max() > 0:
        resp /= resp.max()

    drfb = DRFB_BLEND_HF * hf + DRFB_BLEND_D * resp
    drfb -= drfb.min()
    if drfb.max() > 0:
        drfb /= drfb.max()

    return drfb


def build_stab_gate_from_detections(img_shape,
                                    boxes: List[Tuple[float, float, float, float, float]]) -> np.ndarray:
    H, W = img_shape[:2]
    gate = np.zeros((H, W), dtype=np.float32)

    for (x1, y1, x2, y2, conf) in boxes:
        c = max(0.0, min(float(conf), 1.0))
        if c <= MIN_CONF_WEIGHT:
            continue
        c_norm = (c - MIN_CONF_WEIGHT) / (1.0 - MIN_CONF_WEIGHT)
        c_norm = max(0.0, min(c_norm, 1.0))
        w_conf = c_norm ** STAB_CONF_GAMMA

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 1 or bh <= 1:
            continue

        pad_ratio = 0.4
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        half_w = (0.5 + pad_ratio) * bw
        half_h = (0.5 + pad_ratio) * bh

        xx1 = int(max(0, np.floor(cx - half_w)))
        yy1 = int(max(0, np.floor(cy - half_h)))
        xx2 = int(min(W, np.ceil(cx + half_w)))
        yy2 = int(min(H, np.ceil(cy + half_h)))
        if xx2 <= xx1 or yy2 <= yy1:
            continue

        h_reg = yy2 - yy1
        w_reg = xx2 - xx1

        yy, xx = np.mgrid[0:h_reg, 0:w_reg].astype(np.float32)
        xx = (xx - (w_reg - 1) / 2.0) / (w_reg / 2.0 + 1e-6)
        yy = (yy - (h_reg - 1) / 2.0) / (h_reg / 2.0 + 1e-6)
        r2 = xx * xx + yy * yy
        gauss = np.exp(-r2 / (2.0 * GAUSS_SIGMA * GAUSS_SIGMA))

        patch_gate = w_conf * gauss

        region = gate[yy1:yy2, xx1:xx2]
        np.maximum(region, patch_gate, out=region)
        gate[yy1:yy2, xx1:xx2] = region

    gate = cv2.GaussianBlur(gate, (21, 21), 0)
    if gate.max() > 0:
        gate /= gate.max()

    if STAB_BASE_OUTSIDE > 0.0:
        gate = STAB_BASE_OUTSIDE + (1.0 - STAB_BASE_OUTSIDE) * gate

    return gate


def float_to_heat(m: np.ndarray) -> np.ndarray:
    x = m.copy()
    x -= x.min()
    if x.max() > 0:
        x /= x.max()
    u8 = (x * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    mask = x > 0.04
    out = np.zeros_like(color)
    out[mask] = color[mask]
    return out


def overlay_heat_on_image(img: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Overlay a heat map (float in [0,1]) onto the BGR image with same color coding as float_to_heat.
    """
    heat_color = float_to_heat(heat)
    if heat_color.shape[:2] != img.shape[:2]:
        heat_color = cv2.resize(heat_color, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(img, 1.0 - alpha, heat_color, alpha, 0.0)
    return overlay


# =========================================
# MAIN
# =========================================

def main():
    ROOT_OUT.mkdir(parents=True, exist_ok=True)

    # build ordered list by (clip_id, frame_idx)
    raw_paths = glob.glob(str(IMG_DIR / "*.jpg")) + glob.glob(str(IMG_DIR / "*.png"))
    records = []
    for p in raw_paths:
        img_id = os.path.splitext(os.path.basename(p))[0]
        clip_id, frame_idx = split_clip_frame(img_id)
        records.append((clip_id, frame_idx, p, img_id))
    records.sort(key=lambda r: (r[0], r[1]))

    image_paths = [r[2] for r in records]
    image_ids = [r[3] for r in records]

    print(f"Processing {len(image_ids)} images for metrics")

    frames_ours_better = []
    frames_base_better = []
    frames_both_bad = []

    per_frame_gt = {}
    per_frame_size = {}
    per_frame_path = {}
    per_frame_gt_matched_ours = {}
    per_frame_gt_matched_base = {}

    # first pass: per frame metrics and categories
    for img_path, img_id in tqdm(list(zip(image_paths, image_ids)), total=len(image_ids)):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        gt = load_yolo_boxes_basic(str(LABEL_GT_DIR / f"{img_id}.txt"), has_conf=False)
        ours = load_yolo_boxes_basic(str(PRED_OURS_DIR / f"{img_id}.txt"), has_conf=True)
        base = load_yolo_boxes_basic(str(PRED_BASE_DIR / f"{img_id}.txt"), has_conf=True)

        if not gt and not ours and not base:
            continue

        per_frame_gt[img_id] = gt
        per_frame_size[img_id] = (w, h)
        per_frame_path[img_id] = img_path

        res_o = eval_frame(gt, ours, w, h, CONF_TH_EVAL, IOU_TH_EVAL)
        res_b = eval_frame(gt, base, w, h, CONF_TH_EVAL, IOU_TH_EVAL)

        per_frame_gt_matched_ours[img_id] = res_o["gt_matched"]
        per_frame_gt_matched_base[img_id] = res_b["gt_matched"]

        f1_o = res_o["f1"]
        f1_b = res_b["f1"]

        score_o = f1_o + ALPHA * res_o["mean_iou"]
        score_b = f1_b + ALPHA * res_b["mean_iou"]
        max_score = max(score_o, score_b)

        if f1_o >= GOOD_F1 and score_o >= score_b + MARGIN_SCORE:
            diff = score_o - score_b
            frames_ours_better.append({
                "img_id": img_id,
                "img_path": img_path,
                "diff_score": diff,
                "f1_ours": f1_o,
                "f1_base": f1_b,
                "score_ours": score_o,
                "score_base": score_b,
                "tp_ours": res_o["tp"],
                "fp_ours": res_o["fp"],
                "fn_ours": res_o["fn"],
                "tp_base": res_b["tp"],
                "fp_base": res_b["fp"],
                "fn_base": res_b["fn"],
            })
            continue

        if f1_b >= GOOD_F1 and score_b >= score_o + MARGIN_SCORE:
            diff = score_b - score_o
            frames_base_better.append({
                "img_id": img_id,
                "img_path": img_path,
                "diff_score": diff,
                "f1_ours": f1_o,
                "f1_base": f1_b,
                "score_ours": score_o,
                "score_base": score_b,
                "tp_ours": res_o["tp"],
                "fp_ours": res_o["fp"],
                "fn_ours": res_o["fn"],
                "tp_base": res_b["tp"],
                "fp_base": res_b["fp"],
                "fn_base": res_b["fn"],
            })
            continue

        if max_score <= BAD_F1:
            frames_both_bad.append({
                "img_id": img_id,
                "img_path": img_path,
                "max_score": max_score,
                "f1_ours": f1_o,
                "f1_base": f1_b,
                "score_ours": score_o,
                "score_base": score_b,
                "tp_ours": res_o["tp"],
                "fp_ours": res_o["fp"],
                "fn_ours": res_o["fn"],
                "tp_base": res_b["tp"],
                "fp_base": res_b["fp"],
                "fn_base": res_b["fn"],
            })
            continue

    frames_ours_better.sort(key=lambda d: d["diff_score"], reverse=True)
    frames_base_better.sort(key=lambda d: d["diff_score"], reverse=True)
    frames_both_bad.sort(key=lambda d: d["max_score"])

    print("Finding missed then reappeared objects in consecutive frames")

    ours_missed_then_found = []
    base_missed_then_found = []
    drop_events_ours = []

    for idx in range(len(image_ids) - 1):
        img_id_prev = image_ids[idx]
        img_id_next = image_ids[idx + 1]

        clip_prev, frame_prev = split_clip_frame(img_id_prev)
        clip_next, frame_next = split_clip_frame(img_id_next)

        if clip_prev != clip_next:
            continue

        if frame_prev >= 0 and frame_next >= 0 and frame_next != frame_prev + 1:
            continue

        if img_id_prev not in per_frame_gt or img_id_next not in per_frame_gt:
            continue

        gt_prev = per_frame_gt[img_id_prev]
        gt_next = per_frame_gt[img_id_next]
        if not gt_prev or not gt_next:
            continue

        w_prev, h_prev = per_frame_size[img_id_prev]
        w_next, h_next = per_frame_size[img_id_next]

        matched_prev_ours = per_frame_gt_matched_ours.get(img_id_prev, [])
        matched_next_ours = per_frame_gt_matched_ours.get(img_id_next, [])
        matched_prev_base = per_frame_gt_matched_base.get(img_id_prev, [])
        matched_next_base = per_frame_gt_matched_base.get(img_id_next, [])

        used_next = [False] * len(gt_next)
        for gi_prev, g_prev in enumerate(gt_prev):
            best_iou = 0.0
            best_gj = None

            g_prev_xy = yolo_to_xyxy_dict(g_prev, w_prev, h_prev)

            for gj, g_next in enumerate(gt_next):
                if used_next[gj]:
                    continue
                if g_prev["cls"] != g_next["cls"]:
                    continue
                g_next_xy = yolo_to_xyxy_dict(g_next, w_next, h_next)
                iou_link = iou_xyxy(g_prev_xy, g_next_xy)
                if iou_link > best_iou:
                    best_iou = iou_link
                    best_gj = gj

            if best_gj is None or best_iou < GT_LINK_IOU_TH:
                continue

            used_next[best_gj] = True
            g_next = gt_next[best_gj]
            g_next_xy = yolo_to_xyxy_dict(g_next, w_next, h_next)

            prev_hit_ours = matched_prev_ours[gi_prev] if gi_prev < len(matched_prev_ours) else False
            next_hit_ours = matched_next_ours[best_gj] if best_gj < len(matched_next_ours) else False

            if (not prev_hit_ours) and next_hit_ours:
                ours_missed_then_found.append([
                    "ours",
                    img_id_prev,
                    img_id_next,
                    g_prev["cls"],
                    g_prev_xy[0], g_prev_xy[1], g_prev_xy[2], g_prev_xy[3],
                    g_next_xy[0], g_next_xy[1], g_next_xy[2], g_next_xy[3],
                    best_iou,
                ])

            if prev_hit_ours and (not next_hit_ours):
                drop_events_ours.append({
                    "prev_id": img_id_prev,
                    "curr_id": img_id_next,
                    "cls": g_prev["cls"],
                    "prev_box": g_prev_xy.copy(),
                    "curr_box": g_next_xy.copy(),
                    "gt_iou_link": best_iou,
                })

            prev_hit_base = matched_prev_base[gi_prev] if gi_prev < len(matched_prev_base) else False
            next_hit_base = matched_next_base[best_gj] if best_gj < len(matched_next_base) else False

            if (not prev_hit_base) and next_hit_base:
                base_missed_then_found.append([
                    "baseline",
                    img_id_prev,
                    img_id_next,
                    g_prev["cls"],
                    g_prev_xy[0], g_prev_xy[1], g_prev_xy[2], g_prev_xy[3],
                    g_next_xy[0], g_next_xy[1], g_next_xy[2], g_next_xy[3],
                    best_iou,
                ])

    (ROOT_OUT / "frames").mkdir(exist_ok=True, parents=True)

    with open(ROOT_OUT / "frames" / "frames_ours_better.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "img_id", "img_path", "diff_score",
            "f1_ours", "f1_base", "score_ours", "score_base",
            "tp_ours", "fp_ours", "fn_ours",
            "tp_base", "fp_base", "fn_base",
        ])
        for e in frames_ours_better:
            writer.writerow([
                e["img_id"],
                e["img_path"],
                e["diff_score"],
                e["f1_ours"], e["f1_base"],
                e["score_ours"], e["score_base"],
                e["tp_ours"], e["fp_ours"], e["fn_ours"],
                e["tp_base"], e["fp_base"], e["fn_base"],
            ])

    with open(ROOT_OUT / "frames" / "frames_base_better.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "img_id", "img_path", "diff_score",
            "f1_ours", "f1_base", "score_ours", "score_base",
            "tp_ours", "fp_ours", "fn_ours",
            "tp_base", "fp_base", "fn_base",
        ])
        for e in frames_base_better:
            writer.writerow([
                e["img_id"],
                e["img_path"],
                e["diff_score"],
                e["f1_ours"], e["f1_base"],
                e["score_ours"], e["score_base"],
                e["tp_ours"], e["fp_ours"], e["fn_ours"],
                e["tp_base"], e["fp_base"], e["fn_base"],
            ])

    with open(ROOT_OUT / "frames" / "frames_both_bad.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "img_id", "img_path", "max_score",
            "f1_ours", "f1_base", "score_ours", "score_base",
            "tp_ours", "fp_ours", "fn_ours",
            "tp_base", "fp_base", "fn_base",
        ])
        for e in frames_both_bad:
            writer.writerow([
                e["img_id"],
                e["img_path"],
                e["max_score"],
                e["f1_ours"], e["f1_base"],
                e["score_ours"], e["score_base"],
                e["tp_ours"], e["fp_ours"], e["fn_ours"],
                e["tp_base"], e["fp_base"], e["fn_base"],
            ])

    with open(ROOT_OUT / "frames" / "ours_missed_then_found.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "prev_img_id", "next_img_id",
            "cls",
            "prev_x1", "prev_y1", "prev_x2", "prev_y2",
            "next_x1", "next_y1", "next_x2", "next_y2",
            "gt_iou_link",
        ])
        for row in ours_missed_then_found:
            writer.writerow(row)

    with open(ROOT_OUT / "frames" / "base_missed_then_found.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "prev_img_id", "next_img_id",
            "cls",
            "prev_x1", "prev_y1", "prev_x2", "prev_y2",
            "next_x1", "next_y1", "next_x2", "next_y2",
            "gt_iou_link",
        ])
        for row in base_missed_then_found:
            writer.writerow(row)

    print("Frames where ours is clearly better:", len(frames_ours_better))
    print("Frames where baseline is clearly better:", len(frames_base_better))
    print("Frames where both are bad:", len(frames_both_bad))
    print("Ours missed then found events:", len(ours_missed_then_found))
    print("Baseline missed then found events:", len(base_missed_then_found))
    print("Ours found then missed events:", len(drop_events_ours))

    # =========================================
    # BUILD CATEGORY ID SETS FOR FEATURE MAPS
    # =========================================

    ids_ours_better = {e["img_id"] for e in frames_ours_better}
    ids_base_better = {e["img_id"] for e in frames_base_better}
    ids_both_bad = {e["img_id"] for e in frames_both_bad}

    ids_missed_prev = set()
    ids_missed_next = set()
    for row in ours_missed_then_found:
        _, prev_id, next_id, *_ = row
        ids_missed_prev.add(prev_id)
        ids_missed_next.add(next_id)

    needed_ids = ids_ours_better | ids_base_better | ids_both_bad | ids_missed_prev | ids_missed_next

    fmap_root = ROOT_OUT / "featuremaps"
    fmap_root.mkdir(exist_ok=True, parents=True)

    categories = {
        "ours_better": ids_ours_better,
        "baseline_better": ids_base_better,
        "both_bad": ids_both_bad,
        "missed_prev": ids_missed_prev,
        "missed_next": ids_missed_next,
    }

    cat_dirs = {}
    for name, idset in categories.items():
        cat_base = fmap_root / name
        fpn_dir = cat_base / "fpn"
        drfb_dir = cat_base / "drfb"
        stab_dir = cat_base / "stab"
        for d in [fpn_dir, drfb_dir, stab_dir]:
            d.mkdir(exist_ok=True, parents=True)
        cat_dirs[name] = {
            "ids": idset,
            "fpn": fpn_dir,
            "drfb": drfb_dir,
            "stab": stab_dir,
        }

    # =========================================
    # SECOND PASS: FEATURE MAPS PER CATEGORY
    # =========================================

    print("Second pass: computing feature maps for selected frames")
    prev_drfb_1 = None
    prev_drfb_2 = None
    prev_clip_id = None

    for img_path, img_id in tqdm(list(zip(image_paths, image_ids)), total=len(image_ids)):
        img = cv2.imread(img_path)
        if img is None:
            prev_drfb_2 = prev_drfb_1
            prev_drfb_1 = None
            prev_clip_id = None
            continue

        clip_id, _ = split_clip_frame(img_id)

        if prev_clip_id is not None and clip_id != prev_clip_id:
            prev_drfb_1 = None
            prev_drfb_2 = None
        prev_clip_id = clip_id

        h, w = img.shape[:2]
        boxes = load_yolo_labels_for_fmaps(PRED_OURS_DIR / f"{img_id}.txt", w, h)

        hf = compute_global_hf(img)
        drfb = compute_drfb_response(img, boxes, hf) if boxes else hf.copy()

        neighbors = [drfb]
        if prev_drfb_1 is not None and prev_drfb_1.shape == drfb.shape:
            neighbors.append(prev_drfb_1)
        if prev_drfb_2 is not None and prev_drfb_2.shape == drfb.shape:
            neighbors.append(prev_drfb_2)

        stack = np.stack(neighbors, axis=0)
        stab_base = np.mean(stack, axis=0)
        stab_base = cv2.GaussianBlur(stab_base, (9, 9), 0)

        stab_base = np.power(stab_base, STAB_POW)
        stab_base -= stab_base.min()
        if stab_base.max() > 0:
            stab_base /= stab_base.max()

        gate = build_stab_gate_from_detections(img.shape, boxes)
        stab_resp = stab_base * gate
        stab_resp -= stab_resp.min()
        if stab_resp.max() > 0:
            stab_resp /= stab_resp.max()

        prev_drfb_2 = prev_drfb_1
        prev_drfb_1 = drfb

        if img_id not in needed_ids:
            continue

        heat_fpn = float_to_heat(hf)
        heat_drfb = float_to_heat(drfb)
        heat_stab = float_to_heat(stab_resp)

        for name, info in cat_dirs.items():
            if img_id not in info["ids"]:
                continue
            base_name = img_id
            cv2.imwrite(str(info["fpn"] / f"{base_name}_b_fpn_heat.png"), heat_fpn)
            cv2.imwrite(str(info["drfb"] / f"{base_name}_c_drfb_heat.png"), heat_drfb)
            cv2.imwrite(str(info["stab"] / f"{base_name}_d_drfb_stab_heat.png"), heat_stab)

    # =========================================
    # PANELS FOR Ours missed then found (feature maps only)
    # =========================================

    print("Building DRFB and STAB panels for ours missed then found events")
    out_missed_fmap_dir = ROOT_OUT / "ours_missed_then_found_fmaps"
    out_missed_fmap_dir.mkdir(exist_ok=True, parents=True)

    drfb_prev_dir = fmap_root / "missed_prev" / "drfb"
    stab_prev_dir = fmap_root / "missed_prev" / "stab"
    drfb_next_dir = fmap_root / "missed_next" / "drfb"
    stab_next_dir = fmap_root / "missed_next" / "stab"

    for idx, row in enumerate(tqdm(ours_missed_then_found)):
        _, prev_id, next_id, *_ = row

        prev_drfb_path = drfb_prev_dir / f"{prev_id}_c_drfb_heat.png"
        prev_stab_path = stab_prev_dir / f"{prev_id}_d_drfb_stab_heat.png"
        next_drfb_path = drfb_next_dir / f"{next_id}_c_drfb_heat.png"
        next_stab_path = stab_next_dir / f"{next_id}_d_drfb_stab_heat.png"

        prev_drfb = cv2.imread(str(prev_drfb_path))
        prev_stab = cv2.imread(str(prev_stab_path))
        next_drfb = cv2.imread(str(next_drfb_path))
        next_stab = cv2.imread(str(next_stab_path))

        if prev_drfb is None or prev_stab is None or next_drfb is None or next_stab is None:
            continue

        h, w = prev_drfb.shape[:2]

        def resize_hw(im):
            return cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)

        prev_stab = resize_hw(prev_stab)
        next_drfb = resize_hw(next_drfb)
        next_stab = resize_hw(next_stab)

        top = np.concatenate([prev_drfb, prev_stab], axis=1)
        bottom = np.concatenate([next_drfb, next_stab], axis=1)
        panel = np.concatenate([top, bottom], axis=0)

        out_name = f"{idx:06d}_{prev_id}_{next_id}_drfb_stab.png"
        cv2.imwrite(str(out_missed_fmap_dir / out_name), panel)

    print("Feature map panels built")

    # =========================================
    # NEW: FULL PANELS FOR Ours missed then found
    # two frames with detections plus gt and DRFB STAB overlays
    # =========================================

    print("Building full DRFB STAB panels for ours missed then found events")
    out_missed_full_dir = ROOT_OUT / "ours_missed_then_found_full"
    out_missed_full_dir.mkdir(exist_ok=True, parents=True)

    for idx, row in enumerate(tqdm(ours_missed_then_found)):
        _, prev_id, next_id, cls_id, px1, py1, px2, py2, nx1, ny1, nx2, ny2, link_iou = row

        if prev_id not in per_frame_path or next_id not in per_frame_path:
            continue

        prev_path = per_frame_path[prev_id]
        next_path = per_frame_path[next_id]

        img_prev = cv2.imread(prev_path)
        img_next = cv2.imread(next_path)
        if img_prev is None or img_next is None:
            continue

        hp, wp = img_prev.shape[:2]
        hn, wn = img_next.shape[:2]

        # load gt and detections for both frames
        gt_prev = load_yolo_boxes_basic(str(LABEL_GT_DIR / f"{prev_id}.txt"), has_conf=False)
        gt_next = load_yolo_boxes_basic(str(LABEL_GT_DIR / f"{next_id}.txt"), has_conf=False)

        ours_prev = load_yolo_boxes_basic(str(PRED_OURS_DIR / f"{prev_id}.txt"), has_conf=True)
        ours_next = load_yolo_boxes_basic(str(PRED_OURS_DIR / f"{next_id}.txt"), has_conf=True)

        base_prev = load_yolo_boxes_basic(str(PRED_BASE_DIR / f"{prev_id}.txt"), has_conf=True)
        base_next = load_yolo_boxes_basic(str(PRED_BASE_DIR / f"{next_id}.txt"), has_conf=True)

        # draw detections and gt for prev
        prev_draw = img_prev.copy()
        for g in gt_prev:
            xy = yolo_to_xyxy_dict(g, wp, hp)
            x1, y1, x2, y2 = map(int, xy)
            cv2.rectangle(prev_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for b in base_prev:
            if b["conf"] < CONF_TH_EVAL:
                continue
            xy = yolo_to_xyxy_dict(b, wp, hp)
            x1, y1, x2, y2 = map(int, xy)
            cv2.rectangle(prev_draw, (x1, y1), (x2, y2), (0, 0, 255), 1)

        for o in ours_prev:
            if o["conf"] < CONF_TH_EVAL:
                continue
            xy = yolo_to_xyxy_dict(o, wp, hp)
            x1, y1, x2, y2 = map(int, xy)
            cv2.rectangle(prev_draw, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # highlight linked gt prev box
        lpx1, lpy1, lpx2, lpy2 = map(int, [px1, py1, px2, py2])
        cv2.rectangle(prev_draw, (lpx1, lpy1), (lpx2, lpy2), (0, 255, 255), 3)
        cv2.putText(
            prev_draw,
            "GT linked missed by ours",
            (max(0, lpx1), max(15, lpy1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # draw detections and gt for next
        next_draw = img_next.copy()
        for g in gt_next:
            xy = yolo_to_xyxy_dict(g, wn, hn)
            x1, y1, x2, y2 = map(int, xy)
            cv2.rectangle(next_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for b in base_next:
            if b["conf"] < CONF_TH_EVAL:
                continue
            xy = yolo_to_xyxy_dict(b, wn, hn)
            x1, y1, x2, y2 = map(int, xy)
            cv2.rectangle(next_draw, (x1, y1), (x2, y2), (0, 0, 255), 1)

        for o in ours_next:
            if o["conf"] < CONF_TH_EVAL:
                continue
            xy = yolo_to_xyxy_dict(o, wn, hn)
            x1, y1, x2, y2 = map(int, xy)
            cv2.rectangle(next_draw, (x1, y1), (x2, y2), (255, 0, 0), 1)

        lnx1, lny1, lnx2, lny2 = map(int, [nx1, ny1, nx2, ny2])
        cv2.rectangle(next_draw, (lnx1, lny1), (lnx2, lny2), (0, 255, 255), 3)
        cv2.putText(
            next_draw,
            "GT linked detected by ours",
            (max(0, lnx1), max(15, lny1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # DRFB and STAB for prev and next (using both frames as context)
        boxes_prev_fmap = load_yolo_labels_for_fmaps(PRED_OURS_DIR / f"{prev_id}.txt", wp, hp)
        boxes_next_fmap = load_yolo_labels_for_fmaps(PRED_OURS_DIR / f"{next_id}.txt", wn, hn)

        hf_prev = compute_global_hf(img_prev)
        drfb_prev = compute_drfb_response(img_prev, boxes_prev_fmap, hf_prev) if boxes_prev_fmap else hf_prev.copy()

        hf_next = compute_global_hf(img_next)
        drfb_next = compute_drfb_response(img_next, boxes_next_fmap, hf_next) if boxes_next_fmap else hf_next.copy()

        # STAB for prev
        neighbors_prev = [drfb_prev]
        drfb_next_resize_for_prev = cv2.resize(drfb_next, (wp, hp), interpolation=cv2.INTER_LINEAR)
        neighbors_prev.append(drfb_next_resize_for_prev)
        stack_prev = np.stack(neighbors_prev, axis=0)
        stab_base_prev = np.mean(stack_prev, axis=0)
        stab_base_prev = cv2.GaussianBlur(stab_base_prev, (9, 9), 0)
        stab_base_prev = np.power(stab_base_prev, STAB_POW)
        stab_base_prev -= stab_base_prev.min()
        if stab_base_prev.max() > 0:
            stab_base_prev /= stab_base_prev.max()
        gate_prev = build_stab_gate_from_detections(img_prev.shape, boxes_prev_fmap)
        stab_prev = stab_base_prev * gate_prev
        stab_prev -= stab_prev.min()
        if stab_prev.max() > 0:
            stab_prev /= stab_prev.max()

        # STAB for next
        neighbors_next = [drfb_next]
        drfb_prev_resize_for_next = cv2.resize(drfb_prev, (wn, hn), interpolation=cv2.INTER_LINEAR)
        neighbors_next.append(drfb_prev_resize_for_next)
        stack_next = np.stack(neighbors_next, axis=0)
        stab_base_next = np.mean(stack_next, axis=0)
        stab_base_next = cv2.GaussianBlur(stab_base_next, (9, 9), 0)
        stab_base_next = np.power(stab_base_next, STAB_POW)
        stab_base_next -= stab_base_next.min()
        if stab_base_next.max() > 0:
            stab_base_next /= stab_base_next.max()
        gate_next = build_stab_gate_from_detections(img_next.shape, boxes_next_fmap)
        stab_next = stab_base_next * gate_next
        stab_next -= stab_next.min()
        if stab_next.max() > 0:
            stab_next /= stab_next.max()

        # overlays (DRFB and STAB only)
        prev_drfb_overlay = overlay_heat_on_image(img_prev, drfb_prev)
        prev_stab_overlay = overlay_heat_on_image(img_prev, stab_prev)
        next_drfb_overlay = overlay_heat_on_image(img_next, drfb_next)
        next_stab_overlay = overlay_heat_on_image(img_next, stab_next)

        # ensure shapes for concatenation
        def ensure_hw(im, h_ref, w_ref):
            if im.shape[:2] != (h_ref, w_ref):
                return cv2.resize(im, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)
            return im

        prev_draw = ensure_hw(prev_draw, hp, wp)
        prev_drfb_overlay = ensure_hw(prev_drfb_overlay, hp, wp)
        prev_stab_overlay = ensure_hw(prev_stab_overlay, hp, wp)

        next_draw = ensure_hw(next_draw, hn, wn)
        next_drfb_overlay = ensure_hw(next_drfb_overlay, hn, wn)
        next_stab_overlay = ensure_hw(next_stab_overlay, hn, wn)

        # build panel
        top_row = np.concatenate([prev_draw, prev_drfb_overlay, prev_stab_overlay], axis=1)
        bottom_row = np.concatenate([next_draw, next_drfb_overlay, next_stab_overlay], axis=1)
        panel = np.concatenate([top_row, bottom_row], axis=0)

        out_name = f"{idx:06d}_{prev_id}_{next_id}_missed_found_full.png"
        cv2.imwrite(str(out_missed_full_dir / out_name), panel)

    print("Full missed then found panels saved in", out_missed_full_dir)

    # =========================================
    # PANELS FOR ALL found then missed (ours) already from previous code
    # =========================================

    if drop_events_ours:
        print(f"Building drop example panels for ours found then missed events: {len(drop_events_ours)}")
        out_drop_dir = ROOT_OUT / "drop_example"
        out_drop_dir.mkdir(exist_ok=True, parents=True)

        for idx, drop_event_ours in enumerate(drop_events_ours):
            prev_id = drop_event_ours["prev_id"]
            curr_id = drop_event_ours["curr_id"]

            if curr_id not in per_frame_path:
                continue

            img_curr_path = per_frame_path[curr_id]
            img_prev_path = per_frame_path.get(prev_id, None)

            img_curr = cv2.imread(img_curr_path)
            if img_curr is None:
                continue
            h, w = img_curr.shape[:2]

            gt_curr = load_yolo_boxes_basic(str(LABEL_GT_DIR / f"{curr_id}.txt"), has_conf=False)
            ours_curr = load_yolo_boxes_basic(str(PRED_OURS_DIR / f"{curr_id}.txt"), has_conf=True)
            base_curr = load_yolo_boxes_basic(str(PRED_BASE_DIR / f"{curr_id}.txt"), has_conf=True)

            img_draw = img_curr.copy()

            for g in gt_curr:
                xy = yolo_to_xyxy_dict(g, w, h)
                x1, y1, x2, y2 = map(int, xy)
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for b in base_curr:
                if b["conf"] < CONF_TH_EVAL:
                    continue
                xy = yolo_to_xyxy_dict(b, w, h)
                x1, y1, x2, y2 = map(int, xy)
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 1)

            for o in ours_curr:
                if o["conf"] < CONF_TH_EVAL:
                    continue
                xy = yolo_to_xyxy_dict(o, w, h)
                x1, y1, x2, y2 = map(int, xy)
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), (255, 0, 0), 1)

            curr_box = drop_event_ours["curr_box"]
            cx1, cy1, cx2, cy2 = map(int, curr_box)
            cv2.rectangle(img_draw, (cx1, cy1), (cx2, cy2), (0, 255, 255), 3)
            cv2.putText(
                img_draw,
                "GT linked detected before missed now",
                (max(0, cx1), max(15, cy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            hf_curr = compute_global_hf(img_curr)
            boxes_curr_for_fmap = load_yolo_labels_for_fmaps(PRED_OURS_DIR / f"{curr_id}.txt", w, h)
            drfb_curr = compute_drfb_response(img_curr, boxes_curr_for_fmap, hf_curr) if boxes_curr_for_fmap else hf_curr.copy()

            neighbors = [drfb_curr]
            if img_prev_path is not None:
                img_prev = cv2.imread(img_prev_path)
                if img_prev is not None:
                    hp, wp = img_prev.shape[:2]
                    boxes_prev_for_fmap = load_yolo_labels_for_fmaps(PRED_OURS_DIR / f"{prev_id}.txt", wp, hp)
                    hf_prev = compute_global_hf(img_prev)
                    drfb_prev = compute_drfb_response(img_prev, boxes_prev_for_fmap, hf_prev) if boxes_prev_for_fmap else hf_prev.copy()
                    drfb_prev_resized = cv2.resize(drfb_prev, (w, h), interpolation=cv2.INTER_LINEAR)
                    neighbors.append(drfb_prev_resized)

            stack = np.stack(neighbors, axis=0)
            stab_base = np.mean(stack, axis=0)
            stab_base = cv2.GaussianBlur(stab_base, (9, 9), 0)
            stab_base = np.power(stab_base, STAB_POW)
            stab_base -= stab_base.min()
            if stab_base.max() > 0:
                stab_base /= stab_base.max()

            gate_curr = build_stab_gate_from_detections(img_curr.shape, boxes_curr_for_fmap)
            stab_curr = stab_base * gate_curr
            stab_curr -= stab_curr.min()
            if stab_curr.max() > 0:
                stab_curr /= stab_curr.max()

            fpn_overlay = overlay_heat_on_image(img_curr, hf_curr)
            drfb_overlay = overlay_heat_on_image(img_curr, drfb_curr)
            stab_overlay = overlay_heat_on_image(img_curr, stab_curr)

            def ensure_hw(im):
                if im.shape[:2] != (h, w):
                    return cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
                return im

            img_draw = ensure_hw(img_draw)
            fpn_overlay = ensure_hw(fpn_overlay)
            drfb_overlay = ensure_hw(drfb_overlay)
            stab_overlay = ensure_hw(stab_overlay)

            top_row = np.concatenate([img_draw, fpn_overlay], axis=1)
            bottom_row = np.concatenate([drfb_overlay, stab_overlay], axis=1)
            panel = np.concatenate([top_row, bottom_row], axis=0)

            out_name = f"{idx:06d}_{prev_id}_{curr_id}_drop_panel.png"
            cv2.imwrite(str(out_drop_dir / out_name), panel)

        print("Drop panels saved in", out_drop_dir)
    else:
        print("No found then missed event for ours was found, no drop panels created.")


if __name__ == "__main__":
    main()
