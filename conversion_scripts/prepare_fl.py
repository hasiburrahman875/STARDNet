#!/usr/bin/env python3
"""
End to end NPS to VisDrone to YOLO conversion with consistent names and video length pickles.

Steps
1) Build VisDrone style per frame labels into labels_full from NPS annotations
2) Create temporal half split for each video and symlink frames and labels into
   fl_drones/{train,val}/{frames,labels} while keeping original frame ids in filenames
3) Convert VisDrone labels in each split to YOLO labels into fl_drones/{split}/yolo_labels
4) Write video length pickle files into fl_drones/{split}/videos/video_length_dict.pkl
5) Run simple sanity checks
"""

import os
import glob
import cv2
import shutil
import pickle
from PIL import Image
from tqdm import tqdm
import pandas as pd

# ------------- paths -------------
VIDEOS_DIR   = "/cluster/pixstor/madrias-lab/Hasibur/STARDNet/Datasets/fl/Videos"      # .avi files
FRAMES_DIR   = "/cluster/pixstor/madrias-lab/Hasibur/STARDNet/Datasets/fl/frames"      # Clip_<vid>_<fid>.png with original fid
NPS_ANNO_DIR = "/cluster/pixstor/madrias-lab/Hasibur/STARDNet/Datasets/fl/Annotations" # NPS annotations
OUT_ROOT     = "/cluster/pixstor/madrias-lab/Hasibur/STARDNet/Datasets/fl_drones"      # output root

LABELS_FULL = os.path.join(OUT_ROOT, "labels_full")  # VisDrone labels for all frames
SPLITS = ["train", "val"]


# ------------- utils -------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def list_videos(vid_dir):
    return sorted(glob.glob(os.path.join(vid_dir, "*.avi")))

def get_num_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return nf

def parse_nps_annotation_txt_to_dict(text_file_path):
    """
    NPS line format:
    frame_no, num_obj, x1,y1,x2,y2, x1,y1,x2,y2, ...
    returns dict: { frame_no: [ [x1,y1,x2,y2], ... ] }
    """
    annot = {}
    with open(text_file_path, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(",")
            if len(parts) < 2:
                continue
            frame_no = int(parts[0])
            n_obj = int(parts[1])
            coords = parts[2:]
            boxes = []
            for i in range(0, len(coords), 4):
                if i + 3 >= len(coords):
                    break
                try:
                    x1 = int(coords[i + 0]); y1 = int(coords[i + 1])
                    x2 = int(coords[i + 2]); y2 = int(coords[i + 3])
                    boxes.append([x1, y1, x2, y2])
                except ValueError:
                    continue
            annot[frame_no] = boxes
            if len(boxes) != n_obj:
                # optional warning
                # print(f"count mismatch in {os.path.basename(text_file_path)} frame {frame_no}: {n_obj} vs {len(boxes)}")
                pass
    return annot


# ------------- step 1 -------------
def make_visdrone_labels_full():
    ensure_dir(OUT_ROOT)
    ensure_dir(LABELS_FULL)

    # if per frame labels already exist under NPS_ANNO_DIR copy them
    per_frame_txts = glob.glob(os.path.join(NPS_ANNO_DIR, "Clip_*_*.txt"))
    if per_frame_txts:
        for p in tqdm(per_frame_txts, desc="Copying existing per frame labels to labels_full"):
            dst = os.path.join(LABELS_FULL, os.path.basename(p))
            if not os.path.exists(dst):
                shutil.copy(p, dst)
        return

    # else generate from per video NPS files
    per_video_txts = {os.path.splitext(os.path.basename(p))[0]: p
                      for p in glob.glob(os.path.join(NPS_ANNO_DIR, "Clip_*.txt"))}

    for vid_path in tqdm(list_videos(VIDEOS_DIR), desc="Generating labels_full from NPS"):
        video_base = os.path.splitext(os.path.basename(vid_path))[0]   # Clip_49
        nps_file = per_video_txts.get(video_base)
        if nps_file is None:
            continue

        ann = parse_nps_annotation_txt_to_dict(nps_file)
        for fid, boxes in ann.items():
            out_txt = os.path.join(LABELS_FULL, f"{video_base}_{fid:05d}.txt")
            with open(out_txt, "w") as w:
                for x1, y1, x2, y2 in boxes:
                    w.write(f"{x1},{y1},{x2 - x1},{y2 - y1},1,1,0,0\n")


# ------------- step 2 -------------
def make_half_temporal_split_and_pickles():
    # prepare split dirs
    for split in SPLITS:
        ensure_dir(os.path.join(OUT_ROOT, split, "frames"))
        ensure_dir(os.path.join(OUT_ROOT, split, "labels"))
        ensure_dir(os.path.join(OUT_ROOT, split, "videos"))

    train_video_lens = {}
    val_video_lens = {}

    for vid_path in tqdm(list_videos(VIDEOS_DIR), desc="Creating half temporal split"):
        video_base = os.path.splitext(os.path.basename(vid_path))[0]  # Clip_49
        n_frames = get_num_frames(vid_path)
        mid = n_frames // 2

        # record lengths like your earlier code
        train_video_lens[video_base] = mid
        val_video_lens[video_base] = n_frames - mid

        for split, span in [("train", range(0, mid)), ("val", range(mid, n_frames))]:
            frames_dst = os.path.join(OUT_ROOT, split, "frames")
            labels_dst = os.path.join(OUT_ROOT, split, "labels")
            for fid in span:
                src_img = os.path.join(FRAMES_DIR, f"{video_base}_{fid:05d}.png")
                dst_img = os.path.join(frames_dst, f"{video_base}_{fid:05d}.png")
                if os.path.exists(src_img) and not os.path.exists(dst_img):
                    try:
                        os.symlink(src_img, dst_img)
                    except FileExistsError:
                        pass

                src_lbl = os.path.join(LABELS_FULL, f"{video_base}_{fid:05d}.txt")
                dst_lbl = os.path.join(labels_dst, f"{video_base}_{fid:05d}.txt")
                if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
                    try:
                        os.symlink(src_lbl, dst_lbl)
                    except FileExistsError:
                        pass

    # write pickles like before
    train_pkl = os.path.join(OUT_ROOT, "train", "videos", "video_length_dict.pkl")
    val_pkl   = os.path.join(OUT_ROOT, "val",   "videos", "video_length_dict.pkl")
    with open(train_pkl, "wb") as f:
        pickle.dump(train_video_lens, f)
    with open(val_pkl, "wb") as f:
        pickle.dump(val_video_lens, f)


# ------------- step 3 -------------
def visdrone_bbox_to_yolo(bbox_xywh, img_size):
    w_img, h_img = img_size  # PIL gives (w, h)
    dw = 1.0 / w_img
    dh = 1.0 / h_img
    x, y, w, h = bbox_xywh
    cx = (x + w / 2.0) * dw
    cy = (y + h / 2.0) * dh
    ww = w * dw
    hh = h * dh
    return cx, cy, ww, hh

def convert_split_vis_to_yolo(split):
    vis_labels = os.path.join(OUT_ROOT, split, "labels")
    images_dir = os.path.join(OUT_ROOT, split, "frames")
    yolo_out = os.path.join(OUT_ROOT, split, "yolo_labels")
    ensure_dir(yolo_out)

    files = [f for f in os.listdir(vis_labels) if f.endswith(".txt")]
    print(f"[{split}] total vis labels: {len(files)}")

    for fname in tqdm(files, desc=f"Converting to YOLO for {split}"):
        ann_file = os.path.join(vis_labels, fname)
        img_file = os.path.join(images_dir, fname.replace(".txt", ".png"))
        if not os.path.exists(img_file):
            continue

        try:
            df = pd.read_csv(ann_file, header=None)
        except Exception:
            df = pd.DataFrame()

        try:
            img = Image.open(img_file)
            img_size = img.size  # (w, h)
        except Exception:
            continue

        out_path = os.path.join(yolo_out, fname)
        with open(out_path, "w") as w:
            if df.shape[1] >= 6:
                for _, row in df.iterrows():
                    try:
                        x = float(row[0]); y = float(row[1])
                        ww = float(row[2]); hh = float(row[3])
                        conf = int(row[4])
                        cls  = int(row[5])
                    except Exception:
                        continue
                    if conf == 1 and 0 < cls < 11:
                        cx, cy, nww, nhh = visdrone_bbox_to_yolo((x, y, ww, hh), img_size)
                        w.write(f"{cls - 1} {cx:.6f} {cy:.6f} {nww:.6f} {nhh:.6f}\n")


# ------------- step 4 -------------
def sanity_check_split(split):
    frames_dir = os.path.join(OUT_ROOT, split, "frames")
    labels_dir = os.path.join(OUT_ROOT, split, "labels")
    yolo_dir   = os.path.join(OUT_ROOT, split, "yolo_labels")

    imgs = {os.path.splitext(f)[0] for f in os.listdir(frames_dir) if f.endswith(".png")}
    vis  = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith(".txt")}
    yolo = {os.path.splitext(f)[0] for f in os.listdir(yolo_dir)   if f.endswith(".txt")}

    missing_img_for_vis = sorted(vis - imgs)
    missing_vis_for_img = sorted(imgs - vis)
    missing_yolo        = sorted(imgs - yolo)

    print(f"[{split}] images {len(imgs)} vis {len(vis)} yolo {len(yolo)}")
    print(f"[{split}] vis without image {len(missing_img_for_vis)}")
    print(f"[{split}] image without vis {len(missing_vis_for_img)}")
    print(f"[{split}] frames without yolo {len(missing_yolo)}")


# ------------- main -------------
def main():
    print("Step 1 creating labels_full")
    make_visdrone_labels_full()

    print("Step 2 creating temporal half split and pickles")
    make_half_temporal_split_and_pickles()

    print("Step 3 converting to YOLO")
    for sp in SPLITS:
        convert_split_vis_to_yolo(sp)

    print("Step 4 sanity checks")
    for sp in SPLITS:
        sanity_check_split(sp)

if __name__ == "__main__":
    main()
