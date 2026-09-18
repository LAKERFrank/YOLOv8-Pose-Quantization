#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROI-assisted auto-labeling version:
- Runs pose inference only inside ROI for better detection quality
- BUT saves dataset images as full-frame grayscale letterboxed to 640x640
- Maps ROI detections back to the full-frame 640x640 image coordinate system
- Therefore the output image still keeps the original full image content
- Labels only contain detections that came from the ROI region
- Supports global ROI or per-camera ROI (CR0 / CR1 / ...)
"""

import argparse
import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

SAVE_SIZE = 640
KPT_NUM = 17

_COCO_SKELETON = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


@dataclass
class PoseDet:
    cls: int
    conf: float
    xyxy: np.ndarray       # (4,) float32
    kpts_xy: np.ndarray    # (17,2) float32
    kpts_conf: np.ndarray  # (17,) float32


@dataclass
class VideoJob:
    session_dir: Path
    video_path: Path
    video_name: str


def gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def compute_sample_indices(total_frames: int, src_fps: float, target_fps: float) -> List[int]:
    if total_frames <= 0:
        return []
    if src_fps <= 1e-6:
        return list(range(total_frames))

    duration = total_frames / src_fps
    desired_n = int(duration * target_fps + 0.5)
    if desired_n <= 0:
        desired_n = total_frames

    idxs = []
    last = -1
    for i in range(desired_n):
        t = i / target_fps
        idx = int(t * src_fps)
        if idx >= total_frames:
            idx = total_frames - 1
        if idx != last:
            idxs.append(idx)
            last = idx

    if idxs and idxs[-1] != total_frames - 1:
        idxs.append(total_frames - 1)
    return idxs


def letterbox_params(src_w: int, src_h: int, new_size: int = 640):
    r = min(new_size / float(src_w), new_size / float(src_h))
    new_unpad_w = int(round(src_w * r))
    new_unpad_h = int(round(src_h * r))
    dw = (new_size - new_unpad_w) / 2.0
    dh = (new_size - new_unpad_h) / 2.0
    return r, dw, dh, new_unpad_w, new_unpad_h


def letterbox_gray_with_params(gray: np.ndarray, new_size: int, new_w: int, new_h: int, dw: float, dh: float, pad_val: int = 0):
    src_h, src_w = gray.shape[:2]
    if (new_w, new_h) != (src_w, src_h):
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = gray

    out = np.full((new_size, new_size), pad_val, dtype=np.uint8)
    left = int(round(dw))
    top = int(round(dh))
    out[top:top + new_h, left:left + new_w] = resized
    return out


def run_yolo_pose(model, bgr_img: np.ndarray, imgsz: int, conf: float, iou: float, device: str):
    r = model.predict(
        source=bgr_img,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device if device != "" else None,
        verbose=False,
    )[0]

    if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None or r.keypoints.xy is None:
        return []

    boxes = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    confs = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
    clss = r.boxes.cls.detach().cpu().numpy().astype(np.int32)

    kpts_xy = r.keypoints.xy.detach().cpu().numpy().astype(np.float32)
    kpts_conf = None
    if hasattr(r.keypoints, "conf") and r.keypoints.conf is not None:
        try:
            kpts_conf = r.keypoints.conf.detach().cpu().numpy().astype(np.float32)
        except Exception:
            kpts_conf = None

    dets: List[PoseDet] = []
    for i in range(boxes.shape[0]):
        kc = kpts_conf[i] if kpts_conf is not None else np.ones((KPT_NUM,), dtype=np.float32)
        dets.append(PoseDet(
            cls=int(clss[i]),
            conf=float(confs[i]),
            xyxy=boxes[i],
            kpts_xy=kpts_xy[i],
            kpts_conf=kc,
        ))
    return dets


def clip_xyxy(xyxy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    x1, y1, x2, y2 = xyxy.astype(np.float32).tolist()
    x1 = max(0.0, min(x1, img_w - 1))
    y1 = max(0.0, min(y1, img_h - 1))
    x2 = max(0.0, min(x2, img_w))
    y2 = max(0.0, min(y2, img_h))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def clip_points_xy(pts: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    out = pts.astype(np.float32).copy()
    out[:, 0] = np.clip(out[:, 0], 0.0, float(img_w))
    out[:, 1] = np.clip(out[:, 1], 0.0, float(img_h))
    return out


def map_point_from_lb_to_raw(x: float, y: float, r: float, dw: float, dh: float) -> Tuple[float, float]:
    return (x - dw) / r, (y - dh) / r


def map_point_from_raw_to_lb(x: float, y: float, r: float, dw: float, dh: float) -> Tuple[float, float]:
    return x * r + dw, y * r + dh


def map_roi_det_to_full_frame_lb(
    det: PoseDet,
    rx1: int,
    ry1: int,
    roi_w: int,
    roi_h: int,
    roi_r: float,
    roi_dw: float,
    roi_dh: float,
    full_r: float,
    full_dw: float,
    full_dh: float,
    full_img_w: int,
    full_img_h: int,
) -> PoseDet:
    x1_r, y1_r = map_point_from_lb_to_raw(float(det.xyxy[0]), float(det.xyxy[1]), roi_r, roi_dw, roi_dh)
    x2_r, y2_r = map_point_from_lb_to_raw(float(det.xyxy[2]), float(det.xyxy[3]), roi_r, roi_dw, roi_dh)

    x1_r = np.clip(x1_r, 0.0, float(roi_w))
    y1_r = np.clip(y1_r, 0.0, float(roi_h))
    x2_r = np.clip(x2_r, 0.0, float(roi_w))
    y2_r = np.clip(y2_r, 0.0, float(roi_h))

    x1_full_raw = x1_r + rx1
    y1_full_raw = y1_r + ry1
    x2_full_raw = x2_r + rx1
    y2_full_raw = y2_r + ry1

    x1_full_lb, y1_full_lb = map_point_from_raw_to_lb(x1_full_raw, y1_full_raw, full_r, full_dw, full_dh)
    x2_full_lb, y2_full_lb = map_point_from_raw_to_lb(x2_full_raw, y2_full_raw, full_r, full_dw, full_dh)

    mapped_kpts = np.zeros_like(det.kpts_xy, dtype=np.float32)
    for i in range(det.kpts_xy.shape[0]):
        kx_r, ky_r = map_point_from_lb_to_raw(float(det.kpts_xy[i, 0]), float(det.kpts_xy[i, 1]), roi_r, roi_dw, roi_dh)
        kx_r = np.clip(kx_r, 0.0, float(roi_w))
        ky_r = np.clip(ky_r, 0.0, float(roi_h))
        kx_full_raw = kx_r + rx1
        ky_full_raw = ky_r + ry1
        kx_full_lb, ky_full_lb = map_point_from_raw_to_lb(kx_full_raw, ky_full_raw, full_r, full_dw, full_dh)
        mapped_kpts[i, 0] = kx_full_lb
        mapped_kpts[i, 1] = ky_full_lb

    xyxy = clip_xyxy(np.array([x1_full_lb, y1_full_lb, x2_full_lb, y2_full_lb], dtype=np.float32), full_img_w, full_img_h)
    kpts_xy = clip_points_xy(mapped_kpts, full_img_w, full_img_h)

    return PoseDet(
        cls=det.cls,
        conf=det.conf,
        xyxy=xyxy,
        kpts_xy=kpts_xy,
        kpts_conf=det.kpts_conf.copy() if det.kpts_conf is not None else None,
    )


def write_yolo_pose_label(
    label_path: Path,
    dets: List[PoseDet],
    img_w: int,
    img_h: int,
    kpt_thr: float,
    cls_id: int = 0,
) -> None:
    lines: List[str] = []
    for d in dets:
        x1, y1, x2, y2 = d.xyxy.tolist()
        x1 = max(0.0, min(x1, img_w - 1))
        y1 = max(0.0, min(y1, img_h - 1))
        x2 = max(0.0, min(x2, img_w))
        y2 = max(0.0, min(y2, img_h))

        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw < 2 or bh < 2:
            continue

        xc = x1 + bw * 0.5
        yc = y1 + bh * 0.5

        parts = [
            str(cls_id),
            f"{xc / img_w:.6f}",
            f"{yc / img_h:.6f}",
            f"{bw / img_w:.6f}",
            f"{bh / img_h:.6f}",
        ]

        for j in range(KPT_NUM):
            conf = float(d.kpts_conf[j]) if d.kpts_conf is not None else 1.0
            v = 2 if conf >= kpt_thr else 0
            if v == 0:
                parts += ["0.000000", "0.000000", "0"]
            else:
                x = max(0.0, min(1.0, float(d.kpts_xy[j, 0]) / img_w))
                y = max(0.0, min(1.0, float(d.kpts_xy[j, 1]) / img_h))
                parts += [f"{x:.6f}", f"{y:.6f}", "2"]

        lines.append(" ".join(parts))

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_dataset_yaml(train_dir: Path) -> None:
    yaml_path = train_dir / "data.yaml"
    content = (
        f"path: '{train_dir.resolve().as_posix().replace(chr(39), chr(39) * 2)}'\n"
        "train: images\n"
        "val: images\n"
        "channels: 1\n"
        "names:\n"
        "  0: person\n"
        "kpt_shape: [17, 3]\n"
        "flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n"
    )
    yaml_path.write_text(content, encoding="utf-8")


def _read_yolo_pose_label_txt(label_path: Path):
    if not label_path.exists():
        return []
    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return []

    out = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) < 5 + 17 * 3:
            continue

        cls = int(float(parts[0]))
        xc = float(parts[1]); yc = float(parts[2]); w = float(parts[3]); h = float(parts[4])

        k = []
        base = 5
        for i in range(17):
            x = float(parts[base + i * 3 + 0])
            y = float(parts[base + i * 3 + 1])
            v = int(float(parts[base + i * 3 + 2]))
            k.append((x, y, v))

        out.append({"cls": cls, "xc": xc, "yc": yc, "w": w, "h": h, "kpts": k})
    return out


def _draw_preview_vis(img_gray_path: Path, label_path: Path, out_vis_path: Path, kpt_thr_v: int = 2):
    gray = cv2.imread(str(img_gray_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return

    H, W = gray.shape[:2]
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    items = _read_yolo_pose_label_txt(label_path)
    for it in items:
        xc = it["xc"] * W
        yc = it["yc"] * H
        bw = it["w"] * W
        bh = it["h"] * H
        x1 = int(round(xc - bw / 2))
        y1 = int(round(yc - bh / 2))
        x2 = int(round(xc + bw / 2))
        y2 = int(round(yc + bh / 2))

        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))

        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 128, 0), 2)
        cv2.putText(vis, "person", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2, cv2.LINE_AA)

        kxy = np.zeros((17, 2), dtype=np.float32)
        kv = np.zeros((17,), dtype=np.int32)
        for i, (x, y, v) in enumerate(it["kpts"]):
            kv[i] = v
            kxy[i, 0] = x * W
            kxy[i, 1] = y * H

        for a, b in _COCO_SKELETON:
            if kv[a] >= kpt_thr_v and kv[b] >= kpt_thr_v:
                ax, ay = int(round(kxy[a, 0])), int(round(kxy[a, 1]))
                bx, by = int(round(kxy[b, 0])), int(round(kxy[b, 1]))
                cv2.line(vis, (ax, ay), (bx, by), (0, 255, 255), 2, cv2.LINE_AA)

        for i in range(17):
            if kv[i] >= kpt_thr_v:
                cx, cy = int(round(kxy[i, 0])), int(round(kxy[i, 1]))
                cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1, cv2.LINE_AA)

    out_vis_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_vis_path), vis)


def discover_jobs(root: Path) -> List[VideoJob]:
    jobs: List[VideoJob] = []
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".webm", ".mpg", ".mpeg"}
    for v in sorted(root.rglob("*")):
        if not v.is_file() or v.suffix.lower() not in video_extensions:
            continue
        stem = v.stem
        cr_name = f"CR{stem[len('CameraReader_'):]}" if stem.startswith("CameraReader_") else stem
        jobs.append(VideoJob(session_dir=v.parent, video_path=v, video_name=cr_name))
    return jobs


def parse_roi_str(s: str) -> Tuple[int, int, int, int]:
    vals = [int(v.strip()) for v in s.split(",")]
    if len(vals) != 4:
        raise ValueError(f"ROI must be x1,y1,x2,y2, got: {s}")
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI: {s}")
    return x1, y1, x2, y2


def clamp_roi(roi: Tuple[int, int, int, int], W: int, H: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(x1 + 1, min(W, x2))
    y2 = max(y1 + 1, min(H, y2))
    return x1, y1, x2, y2


def build_roi_map(args) -> Dict[str, Tuple[int, int, int, int]]:
    roi_map: Dict[str, Tuple[int, int, int, int]] = {}
    if args.roi:
        roi_map["default"] = parse_roi_str(args.roi)
    if args.roi_cr0:
        roi_map["CR0"] = parse_roi_str(args.roi_cr0)
    if args.roi_cr1:
        roi_map["CR1"] = parse_roi_str(args.roi_cr1)
    if args.roi_cr2:
        roi_map["CR2"] = parse_roi_str(args.roi_cr2)
    if args.roi_cr3:
        roi_map["CR3"] = parse_roi_str(args.roi_cr3)
    return roi_map


def resolve_roi_for_video(video_name: str, roi_map: Dict[str, Tuple[int, int, int, int]], W: int, H: int) -> Tuple[int, int, int, int]:
    if video_name in roi_map:
        return clamp_roi(roi_map[video_name], W, H)
    if "default" in roi_map:
        return clamp_roi(roi_map["default"], W, H)
    return 0, 0, W, H

def filter_dets_by_missing_kpts(
    dets: List[PoseDet],
    kpt_thr: float,
    max_missing_kpts: int = 4,
) -> List[PoseDet]:
    kept = []
    min_valid_kpts = KPT_NUM - max_missing_kpts

    for d in dets:
        if d.kpts_conf is None:
            continue

        valid_mask = d.kpts_conf >= float(kpt_thr)
        valid_count = int(valid_mask.sum())

        if valid_count >= min_valid_kpts:
            kept.append(d)

    return kept

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset", help="dataset root containing raw-data/; outputs go to labeled-dataset/")
    ap.add_argument("--weights", default="yolov8x-pose-p6.pt", help="pose weights")
    ap.add_argument("--device", default="", help="e.g. 0 / cpu")
    ap.add_argument("--target_fps", type=float, default=120.0)
    ap.add_argument("--imgsz", type=int, default=640, help="inference imgsz")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--kpt_thr", type=float, default=0.15)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--preview_n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--roi", default="", help="global ROI in original frame coords: x1,y1,x2,y2")
    ap.add_argument("--roi_cr0", default="", help="ROI for CR0: x1,y1,x2,y2")
    ap.add_argument("--roi_cr1", default="", help="ROI for CR1: x1,y1,x2,y2")
    ap.add_argument("--roi_cr2", default="", help="ROI for CR2: x1,y1,x2,y2")
    ap.add_argument("--roi_cr3", default="", help="ROI for CR3: x1,y1,x2,y2")

    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    raw_root = root / "raw-data"
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    jobs = discover_jobs(raw_root)
    if not jobs:
        raise RuntimeError(f"No supported videos found in {raw_root}")

    out_train = root / "labeled-dataset"
    out_images = out_train / "images"
    out_labels = out_train / "labels"
    out_preview = out_train / "preview"

    if args.overwrite and out_train.exists():
        shutil.rmtree(out_train)

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    out_preview.mkdir(parents=True, exist_ok=True)

    roi_map = build_roi_map(args)

    from ultralytics import YOLO
    model = YOLO(args.weights)

    print(f"[INFO] root          = {root}")
    print(f"[INFO] out_train     = {out_train}")
    print(f"[INFO] jobs(videos)  = {len(jobs)}")
    print(f"[INFO] target_fps    = {args.target_fps}")
    print(f"[INFO] weights       = {args.weights}")
    print(f"[INFO] save_size     = {SAVE_SIZE}")
    print(f"[INFO] inference     = imgsz={args.imgsz}, conf={args.conf}, iou={args.iou}")
    print(f"[INFO] roi_map       = {roi_map if roi_map else 'FULL FRAME'}")

    produced_pairs: List[Tuple[Path, Path]] = []
    produced_nonempty: List[Tuple[Path, Path]] = []

    for j_i, job in enumerate(jobs, 1):
        cap = cv2.VideoCapture(str(job.video_path))
        if not cap.isOpened():
            print(f"[WARN] cannot open: {job.video_path}")
            continue

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 1e-6:
            src_fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        full_r, full_dw, full_dh, full_new_w, full_new_h = letterbox_params(W, H, SAVE_SIZE)
        rx1, ry1, rx2, ry2 = resolve_roi_for_video(job.video_name, roi_map, W, H)
        roi_w = rx2 - rx1
        roi_h = ry2 - ry1
        roi_r, roi_dw, roi_dh, roi_new_w, roi_new_h = letterbox_params(roi_w, roi_h, SAVE_SIZE)

        sample_idxs = compute_sample_indices(total_frames, float(src_fps), float(args.target_fps))
        eff_fps = (len(sample_idxs) / (total_frames / src_fps)) if total_frames > 0 else 0.0

        print(f"\n[VIDEO {j_i}/{len(jobs)}] {job.video_path}")
        print(f"[INFO] video={job.video_name} res={W}x{H} src_fps={src_fps:.3f} frames={total_frames} sampled={len(sample_idxs)} eff≈{eff_fps:.1f}fps")
        print(f"[INFO] roi={rx1},{ry1},{rx2},{ry2} size={roi_w}x{roi_h}")
        print(f"[INFO] full letterbox r={full_r:.4f} pad=({full_dw:.1f},{full_dh:.1f})")
        print(f"[INFO] roi  letterbox r={roi_r:.4f} pad=({roi_dw:.1f},{roi_dh:.1f})")

        next_ptr = 0
        frame_idx = 0
        kept = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if next_ptr >= len(sample_idxs):
                break
            if frame_idx < sample_idxs[next_ptr]:
                frame_idx += 1
                continue
            if frame_idx != sample_idxs[next_ptr]:
                frame_idx += 1
                continue
            next_ptr += 1

            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_full_lb = letterbox_gray_with_params(
                gray_full,
                new_size=SAVE_SIZE,
                new_w=full_new_w,
                new_h=full_new_h,
                dw=full_dw,
                dh=full_dh,
                pad_val=0,
            )

            roi_bgr = frame[ry1:ry2, rx1:rx2]
            if roi_bgr.size == 0:
                frame_idx += 1
                continue

            gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            gray_roi_lb = letterbox_gray_with_params(
                gray_roi,
                new_size=SAVE_SIZE,
                new_w=roi_new_w,
                new_h=roi_new_h,
                dw=roi_dw,
                dh=roi_dh,
                pad_val=0,
            )

            dets_roi = run_yolo_pose(
                model,
                gray_to_bgr(gray_roi_lb),
                args.imgsz,
                args.conf,
                args.iou,
                args.device,
            )

            dets_roi = filter_dets_by_missing_kpts(
                dets=dets_roi,
                kpt_thr=args.kpt_thr,
                max_missing_kpts=4,
            )

            dets_full = [
                map_roi_det_to_full_frame_lb(
                    det=d,
                    rx1=rx1,
                    ry1=ry1,
                    roi_w=roi_w,
                    roi_h=roi_h,
                    roi_r=roi_r,
                    roi_dw=roi_dw,
                    roi_dh=roi_dh,
                    full_r=full_r,
                    full_dw=full_dw,
                    full_dh=full_dh,
                    full_img_w=SAVE_SIZE,
                    full_img_h=SAVE_SIZE,
                )
                for d in dets_roi
            ]

            session = job.session_dir.name
            video_id = hashlib.sha256(job.video_path.relative_to(raw_root).as_posix().encode("utf-8")).hexdigest()[:16]
            base = f"{session}_{job.video_name}_{video_id}_f{frame_idx:06d}"
            img_path = out_images / f"{base}.png"
            lab_path = out_labels / f"{base}.txt"

            cv2.imwrite(str(img_path), gray_full_lb)

            write_yolo_pose_label(
                label_path=lab_path,
                dets=dets_full,
                img_w=SAVE_SIZE,
                img_h=SAVE_SIZE,
                kpt_thr=args.kpt_thr,
                cls_id=0,
            )

            produced_pairs.append((img_path, lab_path))
            if lab_path.stat().st_size > 0:
                produced_nonempty.append((img_path, lab_path))

            kept += 1
            frame_idx += 1

        cap.release()
        print(f"[INFO] saved frames = {kept}")

    write_dataset_yaml(out_train)

    random.seed(args.seed)
    pool = produced_nonempty if len(produced_nonempty) >= args.preview_n else produced_pairs
    if len(pool) == 0:
        print("[WARN] no outputs generated; preview skipped")
    else:
        n = min(args.preview_n, len(pool))
        picks = random.sample(pool, n)

        for f in out_preview.glob("*"):
            if f.is_file():
                f.unlink()

        for img_path, lab_path in picks:
            dst_img = out_preview / img_path.name
            dst_lab = out_preview / lab_path.name
            shutil.copy2(img_path, dst_img)
            shutil.copy2(lab_path, dst_lab)

            vis_path = out_preview / (img_path.stem + "_vis.png")
            _draw_preview_vis(dst_img, dst_lab, vis_path, kpt_thr_v=2)

        print(f"\n[PREVIEW] wrote {n} pairs (+ vis) to: {out_preview}")

    print("\n[DONE]")
    print(f"  images:  {out_images}")
    print(f"  labels:  {out_labels}")
    print(f"  yaml:    {out_train / 'data.yaml'}")
    print(f"  preview: {out_preview}")


if __name__ == "__main__":
    main()
