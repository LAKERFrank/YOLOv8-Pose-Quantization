import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

try:
    import tensorrt as trt
except Exception as e:
    raise RuntimeError("Failed to import tensorrt. Install TensorRT Python package first.") from e

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
except Exception as e:
    raise RuntimeError("Failed to import pycuda. Install pycuda first.") from e

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception as e:
    raise RuntimeError("Failed to import pycocotools. Install pycocotools first.") from e

try:
    import torch
    from torchvision.ops import nms
except Exception as e:
    raise RuntimeError("Failed to import torch/torchvision; torchvision.ops.nms is required.") from e

try:
    import ultralytics
    from ultralytics.utils import DATASETS_DIR as ULTRA_DATASETS_DIR
except Exception:
    ultralytics = None
    ULTRA_DATASETS_DIR = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LOGGER = trt.Logger(trt.Logger.ERROR)
DEFAULT_KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
DEFAULT_KPT_SIGMAS = np.array(
    [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89],
    dtype=np.float32,
) / 10.0

COMMON_RESULT_FIELDS = [
    "box_map",
    "box_map50",
    "box_map75",
    "pose_map",
    "pose_map50",
    "pose_map75",
    "infer_time_s",
    "fps",
    "speed_preprocess_ms",
    "speed_inference_ms",
    "speed_loss_ms",
    "speed_postprocess_ms",
]

CSV_FIELD_ORDER = [
    "timestamp",
    "dataset",
    "dataset_path",
    "model",
    "model_path",
    "model_kind",
    "eval_batch",
    "engine_input_channels",
    "imgsz",
    "device",
    "status",
    "error",
    *COMMON_RESULT_FIELDS,
    "num_images",
    "num_annotations",
    "preds_json",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate TensorRT YOLOv8 pose engines on YOLO pose datasets")
    p.add_argument("--models", nargs="+", required=True, help="One or more .engine files")
    p.add_argument("--datasets", nargs="+", required=True, help="One or more dataset yaml files or Ultralytics dataset names")
    p.add_argument("--imgsz", type=int, default=640, help="Inference size, square letterbox")
    p.add_argument("--device", type=int, default=0, help="CUDA device index for TensorRT / pycuda")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=300, help="Max detections per image after NMS")
    p.add_argument("--workers", type=int, default=0, help="Unused here; kept for CLI compatibility")
    p.add_argument("--force-batch", type=int, default=None, help="Force eval batch. Default: infer from filename or engine shape")
    p.add_argument("--save-preds-json", action="store_true", help="Save COCO-format predictions JSON per run")
    p.add_argument("--outdir", type=str, default="./val_engine_results", help="Output directory")
    return p.parse_args()


def now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def fmt_num(x: Any, digits: int = 4) -> str:
    if x is None or x == "":
        return ""
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def model_display_name(model_path: str) -> str:
    return Path(model_path).name


def dataset_display_name(dataset_yaml: str) -> str:
    return Path(dataset_yaml).name


def infer_batch_from_name(model_path: str) -> Optional[int]:
    name = Path(model_path).stem.lower()
    m = re.search(r"batch[_-]?(\d+)", name)
    return int(m.group(1)) if m else None


@dataclass
class GTAnnotation:
    bbox_xywh: List[float]
    keypoints: List[float]
    area: float
    num_keypoints: int
    category_id: int = 1
    iscrowd: int = 0


@dataclass
class ImageRecord:
    image_id: int
    file_name: str
    abs_path: str
    width: int
    height: int
    anns: List[GTAnnotation]


def resolve_dataset_yaml_path(yaml_path: str) -> Path:
    p = Path(yaml_path)
    if p.exists():
        return p.resolve()

    # Allow built-in Ultralytics dataset names like coco-pose.yaml
    if ultralytics is not None:
        ultra_root = Path(ultralytics.__file__).resolve().parent
        candidates = [
            ultra_root / "cfg" / "datasets" / yaml_path,
            ultra_root / "yolo" / "data" / "datasets" / yaml_path,
        ]
        for cand in candidates:
            if cand.exists():
                return cand.resolve()

    raise FileNotFoundError(
        f"Dataset yaml not found: {yaml_path}. "
        f"Pass an absolute/local path, or run in an environment where Ultralytics is installed for built-in dataset names."
    )



def read_yaml(path: str) -> Dict[str, Any]:
    yaml_file = resolve_dataset_yaml_path(path)
    with open(yaml_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["__yaml_file__"] = str(yaml_file)
    return cfg



def resolve_dataset_root(cfg: Dict[str, Any], yaml_path: str) -> Path:
    yaml_file = Path(cfg.get("__yaml_file__", yaml_path)).resolve()
    base = yaml_file.parent
    root = cfg.get("path", "")

    if not root:
        return base

    root_path = Path(str(root))
    if root_path.is_absolute():
        return root_path

    candidate = (base / root_path).resolve()
    if candidate.exists():
        return candidate

    # Ultralytics built-in datasets often use path: ../datasets/<name>
    if ULTRA_DATASETS_DIR is not None:
        ultra_candidate = Path(ULTRA_DATASETS_DIR) / root_path.name
        if ultra_candidate.exists():
            return ultra_candidate.resolve()

    return candidate



def resolve_split_spec(root: Path, split_value: Any) -> List[Path]:
    if isinstance(split_value, (list, tuple)):
        files: List[Path] = []
        for item in split_value:
            files.extend(resolve_split_spec(root, item))
        return files

    split_path = Path(str(split_value))
    if not split_path.is_absolute():
        split_path = (root / split_path).resolve()

    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        image_paths: List[Path] = []
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                if not p.is_absolute():
                    p = (root / p).resolve()
                image_paths.append(p)
        return image_paths

    if split_path.is_dir():
        files = [p for p in split_path.rglob("*") if p.suffix.lower() in IMG_EXTS]
        files.sort()
        return files

    if split_path.is_file() and split_path.suffix.lower() in IMG_EXTS:
        return [split_path]

    raise FileNotFoundError(f"Unable to resolve dataset split path: {split_value} -> {split_path}")



def image_to_label_path(img_path: Path) -> Path:
    s = str(img_path)
    s = re.sub(r"([/\\])images([/\\])", r"\1labels\2", s)
    return Path(str(Path(s).with_suffix(".txt")))



def next_image_id_from_path(img_path: Path) -> int:
    stem = img_path.stem
    if stem.isdigit():
        return int(stem)
    return abs(hash(str(img_path))) % (2**31 - 1)



def read_image_shape(img_path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w = img.shape[:2]
    return w, h



def parse_yolo_pose_label(label_path: Path, width: int, height: int, num_kpts: int) -> List[GTAnnotation]:
    anns: List[GTAnnotation] = []
    if not label_path.exists():
        return anns

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return anns

    for line in text.splitlines():
        vals = [float(x) for x in line.strip().split()]
        if len(vals) < 5:
            continue

        cls = int(vals[0])
        xc, yc, bw, bh = vals[1:5]
        x = (xc - bw / 2.0) * width
        y = (yc - bh / 2.0) * height
        w = bw * width
        h = bh * height
        bbox_xywh = [x, y, w, h]

        kpt_vals = vals[5:]
        keypoints: List[float] = []
        num_keypoints = 0
        for i in range(num_kpts):
            base = i * 3
            if base + 2 >= len(kpt_vals):
                keypoints.extend([0.0, 0.0, 0.0])
                continue
            kx = kpt_vals[base] * width
            ky = kpt_vals[base + 1] * height
            kv = kpt_vals[base + 2]
            if kv > 0:
                num_keypoints += 1
            keypoints.extend([kx, ky, kv])

        anns.append(
            GTAnnotation(
                bbox_xywh=bbox_xywh,
                keypoints=keypoints,
                area=max(w, 0.0) * max(h, 0.0),
                num_keypoints=num_keypoints,
                category_id=cls + 1,
            )
        )
    return anns



def load_yolo_pose_dataset(yaml_path: str, split: str = "val") -> Tuple[List[ImageRecord], Dict[str, Any]]:
    cfg = read_yaml(yaml_path)
    root = resolve_dataset_root(cfg, yaml_path)
    split_value = cfg.get(split)
    if split_value is None:
        raise KeyError(f"Dataset yaml missing split '{split}': {yaml_path}")

    kpt_shape = cfg.get("kpt_shape", [17, 3])
    num_kpts = int(kpt_shape[0])
    names = cfg.get("names", {0: "person"})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    image_paths = resolve_split_spec(root, split_value)
    records: List[ImageRecord] = []

    for img_path in image_paths:
        w, h = read_image_shape(img_path)
        label_path = image_to_label_path(img_path)
        anns = parse_yolo_pose_label(label_path, w, h, num_kpts=num_kpts)
        records.append(
            ImageRecord(
                image_id=next_image_id_from_path(img_path),
                file_name=img_path.name,
                abs_path=str(img_path),
                width=w,
                height=h,
                anns=anns,
            )
        )

    meta = {
        "root": str(root),
        "num_kpts": num_kpts,
        "kpt_names": DEFAULT_KPT_NAMES[:num_kpts],
        "names": names,
    }
    return records, meta



def build_coco_gt_dict(records: List[ImageRecord], meta: Dict[str, Any]) -> Dict[str, Any]:
    images = []
    annotations = []
    ann_id = 1

    for rec in records:
        images.append({
            "id": rec.image_id,
            "file_name": rec.file_name,
            "width": rec.width,
            "height": rec.height,
        })
        for ann in rec.anns:
            annotations.append({
                "id": ann_id,
                "image_id": rec.image_id,
                "category_id": ann.category_id,
                "bbox": ann.bbox_xywh,
                "area": ann.area,
                "iscrowd": ann.iscrowd,
                "num_keypoints": ann.num_keypoints,
                "keypoints": ann.keypoints,
            })
            ann_id += 1

    categories = [{
        "id": 1,
        "name": meta["names"].get(0, "person"),
        "supercategory": "person",
        "keypoints": meta["kpt_names"],
        "skeleton": [],
    }]

    return {
        "info": {"description": "YOLO-pose dataset converted to COCO in memory"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }



def make_coco_api(coco_dict: Dict[str, Any]) -> COCO:
    coco = COCO()
    coco.dataset = coco_dict
    coco.createIndex()
    return coco


class TensorRTEngine:
    def __init__(self, engine_path: str, device_id: int = 0):
        self.engine_path = str(engine_path)
        self.device_id = int(device_id)
        self.runtime = trt.Runtime(LOGGER)
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create execution context: {engine_path}")
        self.stream = cuda.Stream()

        self.use_v3 = hasattr(self.engine, "num_io_tensors")
        if self.use_v3:
            self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.input_names = [n for n in self.tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            self.output_names = [n for n in self.tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            if len(self.input_names) != 1:
                raise RuntimeError(f"Expected exactly 1 input tensor, got {self.input_names}")
            self.input_name = self.input_names[0]
            self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
            self.engine_input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        else:
            self.num_bindings = self.engine.num_bindings
            self.input_names = [self.engine.get_binding_name(i) for i in range(self.num_bindings) if self.engine.binding_is_input(i)]
            self.output_names = [self.engine.get_binding_name(i) for i in range(self.num_bindings) if not self.engine.binding_is_input(i)]
            if len(self.input_names) != 1:
                raise RuntimeError(f"Expected exactly 1 input binding, got {self.input_names}")
            self.input_name = self.input_names[0]
            idx = self.engine.get_binding_index(self.input_name)
            self.input_dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            self.engine_input_shape = tuple(self.engine.get_binding_shape(idx))

        self.fixed_batch_hint = self._infer_fixed_batch_hint()
        self.input_channels_hint = self._infer_input_channels_hint()

    def _infer_fixed_batch_hint(self) -> Optional[int]:
        shape = self.engine_input_shape
        if len(shape) >= 1 and shape[0] > 0:
            return int(shape[0])
        return None

    def _infer_input_channels_hint(self) -> int:
        shape = self.engine_input_shape
        if len(shape) >= 2 and shape[1] > 0:
            return int(shape[1])
        return 3

    def _set_input_shape(self, shape: Tuple[int, int, int, int]) -> None:
        if self.use_v3:
            self.context.set_input_shape(self.input_name, shape)
        else:
            idx = self.engine.get_binding_index(self.input_name)
            self.context.set_binding_shape(idx, shape)

    def _allocate_buffers_v3(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        host_outs: Dict[str, np.ndarray] = {}
        dev_ptrs: Dict[str, Any] = {}
        for name in self.tensor_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if np.prod(shape) < 0:
                raise RuntimeError(f"Dynamic tensor shape not resolved for {name}: {shape}")
            host = np.empty(shape, dtype=dtype)
            dev = cuda.mem_alloc(host.nbytes)
            dev_ptrs[name] = dev
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                host_outs[name] = host
            self.context.set_tensor_address(name, int(dev))
        return host_outs, dev_ptrs

    def _allocate_buffers_v2(self) -> Tuple[Dict[str, np.ndarray], List[int], Dict[int, Any]]:
        bindings: List[int] = [0] * self.num_bindings
        host_outs: Dict[str, np.ndarray] = {}
        dev_ptrs_by_index: Dict[int, Any] = {}
        for i in range(self.num_bindings):
            shape = tuple(self.context.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            if np.prod(shape) < 0:
                raise RuntimeError(f"Dynamic binding shape not resolved for index {i}: {shape}")
            host = np.empty(shape, dtype=dtype)
            dev = cuda.mem_alloc(host.nbytes)
            bindings[i] = int(dev)
            dev_ptrs_by_index[i] = dev
            if not self.engine.binding_is_input(i):
                host_outs[self.engine.get_binding_name(i)] = host
        return host_outs, bindings, dev_ptrs_by_index

    def infer(self, x: np.ndarray, requested_batch: Optional[int] = None) -> Dict[str, np.ndarray]:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW input, got shape={x.shape}")

        actual_bs = int(x.shape[0])
        run_bs = requested_batch or actual_bs
        if self.fixed_batch_hint is not None and self.fixed_batch_hint > 0 and self.fixed_batch_hint != actual_bs:
            run_bs = self.fixed_batch_hint
            if actual_bs > run_bs:
                raise RuntimeError(f"Input batch {actual_bs} exceeds engine fixed batch {run_bs}")
            if actual_bs < run_bs:
                pad_count = run_bs - actual_bs
                pad = np.repeat(x[-1:], pad_count, axis=0)
                x = np.concatenate([x, pad], axis=0)

        x = np.ascontiguousarray(x.astype(self.input_dtype, copy=False))
        self._set_input_shape(tuple(x.shape))

        if self.use_v3:
            host_outs, dev_ptrs = self._allocate_buffers_v3()
            cuda.memcpy_htod_async(dev_ptrs[self.input_name], x, self.stream)
            ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3() failed")
            for name in self.output_names:
                cuda.memcpy_dtoh_async(host_outs[name], dev_ptrs[name], self.stream)
            self.stream.synchronize()
            for name, arr in list(host_outs.items()):
                if arr.ndim >= 1 and arr.shape[0] >= actual_bs:
                    host_outs[name] = arr[:actual_bs].copy()
            return host_outs

        host_outs, bindings, dev_ptrs = self._allocate_buffers_v2()
        in_idx = self.engine.get_binding_index(self.input_name)
        cuda.memcpy_htod_async(dev_ptrs[in_idx], x, self.stream)
        ok = self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2() failed")
        for name, host in host_outs.items():
            out_idx = self.engine.get_binding_index(name)
            cuda.memcpy_dtoh_async(host, dev_ptrs[out_idx], self.stream)
        self.stream.synchronize()
        for name, arr in list(host_outs.items()):
            if arr.ndim >= 1 and arr.shape[0] >= actual_bs:
                host_outs[name] = arr[:actual_bs].copy()
        return host_outs



def letterbox(im: np.ndarray, new_shape: int = 640, color: Any = 114) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)



def preprocess_image(img_path: str, imgsz: int, input_channels: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    im0 = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if im0 is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    if input_channels == 1:
        if im0.ndim == 3:
            if im0.shape[2] == 4:
                im0 = cv2.cvtColor(im0, cv2.COLOR_BGRA2GRAY)
            else:
                im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        h0, w0 = im0.shape[:2]
        im, ratio, (dw, dh) = letterbox(im0, new_shape=imgsz, color=114)
        im = im[None, :, :]
    elif input_channels == 3:
        if im0.ndim == 2:
            im0 = cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)
        elif im0.shape[2] == 4:
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGRA2BGR)
        h0, w0 = im0.shape[:2]
        im, ratio, (dw, dh) = letterbox(im0, new_shape=imgsz, color=(114, 114, 114))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.transpose(2, 0, 1)
    else:
        raise RuntimeError(f"Unsupported engine input channels: {input_channels}")

    im = np.ascontiguousarray(im, dtype=np.float32) / 255.0
    info = {
        "orig_shape": (h0, w0),
        "ratio": ratio,
        "pad": (dw, dh),
        "img_path": str(img_path),
    }
    return im, info



def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y



def scale_boxes_xyxy(boxes: np.ndarray, orig_shape: Tuple[int, int], ratio: float, pad: Tuple[float, float]) -> np.ndarray:
    boxes = boxes.copy()
    dw, dh = pad
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= ratio
    h, w = orig_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes



def scale_keypoints(kpts: np.ndarray, orig_shape: Tuple[int, int], ratio: float, pad: Tuple[float, float]) -> np.ndarray:
    kpts = kpts.copy()
    dw, dh = pad
    kpts[..., 0] -= dw
    kpts[..., 1] -= dh
    kpts[..., :2] /= ratio
    h, w = orig_shape
    kpts[..., 0] = kpts[..., 0].clip(0, w)
    kpts[..., 1] = kpts[..., 1].clip(0, h)
    return kpts



def squeeze_output_to_bna(arr: np.ndarray, num_kpts: int = 17) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise RuntimeError(f"Unsupported output ndim={arr.ndim}, shape={arr.shape}")

    a = arr.shape[-1]
    b = arr.shape[1]
    min_attrs = 4 + 1 + num_kpts * 3
    if a >= min_attrs and a <= 256:
        return arr
    if b >= min_attrs and b <= 256:
        return np.transpose(arr, (0, 2, 1))
    raise RuntimeError(f"Unable to infer output layout for shape={arr.shape}")



def pick_main_output(outputs: Dict[str, np.ndarray]) -> np.ndarray:
    items = sorted(outputs.items(), key=lambda kv: np.prod(kv[1].shape), reverse=True)
    if not items:
        raise RuntimeError("No output tensors from TensorRT engine")
    return items[0][1]



def decode_yolov8_pose_batch(
    main_output: np.ndarray,
    infos: List[Dict[str, Any]],
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    num_kpts: int = 17,
) -> List[Dict[str, np.ndarray]]:
    pred = squeeze_output_to_bna(main_output, num_kpts=num_kpts)
    attrs = pred.shape[-1]
    nc = attrs - 4 - num_kpts * 3
    if nc <= 0:
        raise RuntimeError(f"Invalid attrs={attrs}; computed nc={nc} for num_kpts={num_kpts}")

    results: List[Dict[str, np.ndarray]] = []
    for bi in range(pred.shape[0]):
        x = pred[bi]
        boxes_xywh = x[:, :4]
        cls_scores = x[:, 4:4 + nc]
        best_cls = cls_scores.argmax(axis=1)
        best_conf = cls_scores.max(axis=1)
        kpts = x[:, 4 + nc:].reshape(-1, num_kpts, 3)

        keep = best_conf > conf_thres
        boxes_xywh = boxes_xywh[keep]
        best_conf = best_conf[keep]
        best_cls = best_cls[keep]
        kpts = kpts[keep]

        if boxes_xywh.shape[0] == 0:
            results.append({
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "classes": np.zeros((0,), dtype=np.int32),
                "kpts": np.zeros((0, num_kpts, 3), dtype=np.float32),
            })
            continue

        boxes_xyxy = xywh2xyxy(boxes_xywh.astype(np.float32))
        keep_idx = nms(
            torch.from_numpy(boxes_xyxy),
            torch.from_numpy(best_conf.astype(np.float32)),
            iou_thres,
        ).cpu().numpy()
        if max_det > 0:
            keep_idx = keep_idx[:max_det]

        boxes_xyxy = boxes_xyxy[keep_idx]
        scores = best_conf[keep_idx].astype(np.float32)
        classes = best_cls[keep_idx].astype(np.int32)
        kpts = kpts[keep_idx].astype(np.float32)

        info = infos[bi]
        boxes_xyxy = scale_boxes_xyxy(boxes_xyxy, info["orig_shape"], info["ratio"], info["pad"])
        kpts = scale_keypoints(kpts, info["orig_shape"], info["ratio"], info["pad"])

        results.append({
            "boxes": boxes_xyxy,
            "scores": scores,
            "classes": classes,
            "kpts": kpts,
        })
    return results



def coco_results_from_batch_preds(batch_preds: List[Dict[str, np.ndarray]], batch_records: List[ImageRecord], category_id: int = 1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    bbox_results: List[Dict[str, Any]] = []
    kpt_results: List[Dict[str, Any]] = []
    for preds, rec in zip(batch_preds, batch_records):
        boxes = preds["boxes"]
        scores = preds["scores"]
        kpts = preds["kpts"]
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            score = float(scores[i])
            bbox_results.append({
                "image_id": rec.image_id,
                "category_id": category_id,
                "bbox": [x1, y1, w, h],
                "score": score,
            })
            flat_kpts: List[float] = []
            for kp in kpts[i].tolist():
                flat_kpts.extend([float(kp[0]), float(kp[1]), float(max(kp[2], 0.0))])
            kpt_results.append({
                "image_id": rec.image_id,
                "category_id": category_id,
                "keypoints": flat_kpts,
                "score": score,
            })
    return bbox_results, kpt_results



def summarize_cocoeval(coco_gt: COCO, preds: List[Dict[str, Any]], iou_type: str, sigmas: Optional[np.ndarray] = None) -> Dict[str, Optional[float]]:
    if len(preds) == 0:
        return {"map": 0.0, "map50": 0.0, "map75": 0.0}
    coco_dt = coco_gt.loadRes(preds)
    evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    if iou_type == "keypoints" and sigmas is not None:
        evaluator.params.kpt_oks_sigmas = np.array(sigmas, dtype=np.float32)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    return {"map": safe_float(stats[0]), "map50": safe_float(stats[1]), "map75": safe_float(stats[2])}



def evaluate_engine_on_dataset(
    engine_path: str,
    dataset_yaml: str,
    imgsz: int,
    device_id: int,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    force_batch: Optional[int],
    save_preds_json: bool,
    outdir: Path,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "timestamp": now_ts(),
        "dataset": dataset_display_name(dataset_yaml),
        "dataset_path": str(dataset_yaml),
        "model": model_display_name(engine_path),
        "model_path": str(engine_path),
        "model_kind": "engine",
        "eval_batch": None,
        "engine_input_channels": None,
        "imgsz": imgsz,
        "device": device_id,
        "status": "ok",
        "error": "",
        "box_map": None,
        "box_map50": None,
        "box_map75": None,
        "pose_map": None,
        "pose_map50": None,
        "pose_map75": None,
        "infer_time_s": None,
        "fps": None,
        "speed_preprocess_ms": None,
        "speed_inference_ms": None,
        "speed_loss_ms": None,
        "speed_postprocess_ms": None,
        "num_images": None,
        "num_annotations": None,
        "preds_json": "",
    }

    try:
        records, meta = load_yolo_pose_dataset(dataset_yaml, split="val")
        row["num_images"] = len(records)
        row["num_annotations"] = sum(len(r.anns) for r in records)

        coco_gt = make_coco_api(build_coco_gt_dict(records, meta))

        engine = TensorRTEngine(engine_path, device_id=device_id)
        row["engine_input_channels"] = engine.input_channels_hint
        eval_batch = force_batch or infer_batch_from_name(engine_path) or engine.fixed_batch_hint or 1
        row["eval_batch"] = int(eval_batch)

        bbox_preds_all: List[Dict[str, Any]] = []
        kpt_preds_all: List[Dict[str, Any]] = []
        total_preprocess_s = 0.0
        total_infer_s = 0.0
        total_postprocess_s = 0.0

        for start in range(0, len(records), eval_batch):
            batch_records = records[start:start + eval_batch]
            ims = []
            infos = []
            t_pre0 = time.perf_counter()
            for rec in batch_records:
                im, info = preprocess_image(rec.abs_path, imgsz=imgsz, input_channels=engine.input_channels_hint)
                ims.append(im)
                infos.append(info)
            x = np.stack(ims, axis=0)
            total_preprocess_s += time.perf_counter() - t_pre0

            t_inf0 = time.perf_counter()
            outputs = engine.infer(x, requested_batch=eval_batch)
            total_infer_s += time.perf_counter() - t_inf0

            t_post0 = time.perf_counter()
            main_output = pick_main_output(outputs)
            batch_preds = decode_yolov8_pose_batch(
                main_output=main_output,
                infos=infos,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
                num_kpts=meta["num_kpts"],
            )
            bbox_results, kpt_results = coco_results_from_batch_preds(batch_preds, batch_records, category_id=1)
            bbox_preds_all.extend(bbox_results)
            kpt_preds_all.extend(kpt_results)
            total_postprocess_s += time.perf_counter() - t_post0

        row["infer_time_s"] = total_infer_s
        row["fps"] = (len(records) / total_infer_s) if total_infer_s > 0 else None
        if len(records) > 0:
            row["speed_preprocess_ms"] = total_preprocess_s * 1000.0 / len(records)
            row["speed_inference_ms"] = total_infer_s * 1000.0 / len(records)
            row["speed_loss_ms"] = None
            row["speed_postprocess_ms"] = total_postprocess_s * 1000.0 / len(records)

        box_stats = summarize_cocoeval(coco_gt, bbox_preds_all, iou_type="bbox")
        pose_stats = summarize_cocoeval(coco_gt, kpt_preds_all, iou_type="keypoints", sigmas=DEFAULT_KPT_SIGMAS[:meta["num_kpts"]])
        row["box_map"] = box_stats["map"]
        row["box_map50"] = box_stats["map50"]
        row["box_map75"] = box_stats["map75"]
        row["pose_map"] = pose_stats["map"]
        row["pose_map50"] = pose_stats["map50"]
        row["pose_map75"] = pose_stats["map75"]

        if save_preds_json:
            stem = f"{Path(engine_path).stem}__{Path(dataset_yaml).stem}"
            preds_path = outdir / f"{stem}.preds.json"
            payload = {"bbox": bbox_preds_all, "keypoints": kpt_preds_all}
            preds_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            row["preds_json"] = str(preds_path)

    except Exception as e:
        row["status"] = "fail"
        row["error"] = f"{type(e).__name__}: {e}"

    return row



def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    if not rows:
        return
    keys = [k for k in CSV_FIELD_ORDER if k in rows[0]]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)



def write_json(rows: List[Dict[str, Any]], out_json: Path) -> None:
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")



def rows_to_markdown(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "# TensorRT Pose Validation\n\nNo results.\n"
    headers = [
        "dataset",
        "model",
        "eval_batch",
        "engine_input_channels",
        *COMMON_RESULT_FIELDS,
        "status",
    ]
    lines = ["# TensorRT Pose Validation\n", f"Generated at: {now_ts()}\n"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h, "")
            if h in set(COMMON_RESULT_FIELDS):
                vals.append(fmt_num(v))
            else:
                vals.append(str(v) if v is not None else "")
        lines.append("| " + " | ".join(vals) + " |")
    fails = [r for r in rows if r.get("status") != "ok"]
    if fails:
        lines.append("\n## Failed Runs\n")
        for r in fails:
            lines.append(f"- dataset=`{r['dataset']}` model=`{r['model']}` error=`{r['error']}`")
    lines.append("")
    return "\n".join(lines)



def print_console_summary(rows: List[Dict[str, Any]]) -> None:
    print("\n========== TensorRT Validation Summary ==========")
    for r in rows:
        if r["status"] == "ok":
            print(
                f"[OK]   dataset={r['dataset']:<20} "
                f"model={r['model']:<25} "
                f"batch={str(r['eval_batch']):<4} "
                f"ch={str(r['engine_input_channels']):<2} "
                f"pose_map={fmt_num(r['pose_map'])} "
                f"pose_map50={fmt_num(r['pose_map50'])} "
                f"fps={fmt_num(r['fps'])}"
            )
        else:
            print(
                f"[FAIL] dataset={r['dataset']:<20} "
                f"model={r['model']:<25} "
                f"batch={str(r.get('eval_batch', '')):<4} "
                f"ch={str(r.get('engine_input_channels', '')):<2} "
                f"error={r['error']}"
            )
    print("=================================================\n")



def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    total = len(args.models) * len(args.datasets)
    idx = 0

    print("Models:")
    for m in args.models:
        print(f"  - {m}")
    print("Datasets:")
    for d in args.datasets:
        print(f"  - {d}")
    print("")

    for dataset_yaml in args.datasets:
        for engine_path in args.models:
            idx += 1
            print(f"[{idx}/{total}] Validating engine={engine_path} on dataset={dataset_yaml}")
            row = evaluate_engine_on_dataset(
                engine_path=engine_path,
                dataset_yaml=dataset_yaml,
                imgsz=args.imgsz,
                device_id=args.device,
                conf_thres=args.conf,
                iou_thres=args.iou,
                max_det=args.max_det,
                force_batch=args.force_batch,
                save_preds_json=args.save_preds_json,
                outdir=outdir,
            )
            rows.append(row)

    out_csv = outdir / "results.csv"
    out_json = outdir / "results.json"
    out_md = outdir / "results.md"
    write_csv(rows, out_csv)
    write_json(rows, out_json)
    out_md.write_text(rows_to_markdown(rows), encoding="utf-8")
    print_console_summary(rows)
    print(f"Saved CSV : {out_csv}")
    print(f"Saved JSON: {out_json}")
    print(f"Saved MD  : {out_md}")

    if any(r["status"] != "ok" for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()