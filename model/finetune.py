#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tune a TRUE 1-channel grayscale YOLOv8-Pose model.
"""

import argparse
from typing import Optional, Tuple

import torch
import torch.nn as nn
from ultralytics import YOLO


def find_first_conv(net: nn.Module) -> Tuple[Optional[str], Optional[nn.Conv2d]]:
    """
    Return the first Conv2d module in the model.
    """
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            return name, module
    return None, None


def replace_module_by_name(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """
    Replace a submodule by dotted path name.
    Supports Sequential / ModuleList numeric child names.
    """
    if "." in module_name:
        parent_path, child_name = module_name.rsplit(".", 1)
        parent = root.get_submodule(parent_path)
    else:
        parent = root
        child_name = module_name

    if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = new_module
    else:
        setattr(parent, child_name, new_module)


def patch_first_conv_to_1ch_if_needed(yolo: YOLO) -> None:
    """
    Safety patch:
    - If model is already 1ch, do nothing.
    - If model is 3ch, convert first conv 3->1 by averaging RGB weights.
    """
    net = yolo.model
    first_name, first_conv = find_first_conv(net)
    if first_conv is None or first_name is None:
        raise RuntimeError("Cannot find first Conv2d in model.")

    if first_conv.in_channels == 1:
        print(f"[OK] first conv already 1ch: {first_name}")
        return

    if first_conv.in_channels != 3:
        raise RuntimeError(
            f"Unexpected first conv in_channels={first_conv.in_channels}, expected 1 or 3."
        )

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=(first_conv.bias is not None),
        padding_mode=first_conv.padding_mode,
    ).to(device=first_conv.weight.device, dtype=first_conv.weight.dtype)

    with torch.no_grad():
        new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
        if first_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias)

    replace_module_by_name(net, first_name, new_conv)
    print(f"[PATCH] first conv converted 3ch -> 1ch: {first_name}")


def verify_first_conv(yolo: YOLO) -> None:
    net = yolo.model
    first_name, first_conv = find_first_conv(net)
    if first_conv is None or first_name is None:
        print("[WARN] no Conv2d found")
        return
    print(
        f"[VERIFY] first conv = {first_name}, "
        f"in_channels = {first_conv.in_channels}, "
        f"out_channels = {first_conv.out_channels}"
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    # Core finetune args
    ap.add_argument("--data", required=True, help="dataset.yaml path (should contain channels: 1)")
    ap.add_argument("--weights", required=True, help="path to previous best.pt / last.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="e.g. 0 / 0,1 / cpu")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--project", default="finetune")
    ap.add_argument("--name", default="counterclockwise-invert-100")

    # Finetune LR / freezing
    ap.add_argument("--lr0", type=float, default=0.001, help="initial learning rate for finetune")
    ap.add_argument("--lrf", type=float, default=0.01, help="final lr factor")
    ap.add_argument("--freeze", type=int, default=0, help="freeze first N layers; 0 means no freeze")

    # Pose-friendly conservative augmentation
    ap.add_argument("--degrees", type=float, default=2.0, help="small rotation only")
    ap.add_argument("--translate", type=float, default=0.02, help="small translation only")
    ap.add_argument("--scale", type=float, default=0.15, help="small scale jitter")
    ap.add_argument("--shear", type=float, default=0.0, help="disable shear for pose stability")
    ap.add_argument("--perspective", type=float, default=0.0, help="disable perspective for pose stability")
    ap.add_argument("--flipud", type=float, default=0.0, help="usually disable vertical flip for human pose")
    ap.add_argument(
        "--fliplr",
        type=float,
        default=0.5,
        help="left-right flip probability; set 0.0 if your flip_idx is not correct",
    )
    ap.add_argument(
        "--mosaic",
        type=float,
        default=0.0,
        help="default 0.0 for safer keypoint finetuning; try 0.1~0.2 only if needed",
    )
    ap.add_argument(
        "--close-mosaic",
        dest="close_mosaic",
        type=int,
        default=10,
        help="has effect only when mosaic > 0",
    )

    return ap


def main() -> None:
    args = build_parser().parse_args()

    # Load previous trained weight
    yolo = YOLO(args.weights)

    # Safety: ensure true 1ch model
    patch_first_conv_to_1ch_if_needed(yolo)
    verify_first_conv(yolo)

    print("[INFO] Finetuning with conservative pose augmentation:")
    print(f"  degrees      = {args.degrees}")
    print(f"  translate    = {args.translate}")
    print(f"  scale        = {args.scale}")
    print(f"  shear        = {args.shear}")
    print(f"  perspective  = {args.perspective}")
    print(f"  flipud       = {args.flipud}")
    print(f"  fliplr       = {args.fliplr}")
    print(f"  mosaic       = {args.mosaic}")
    print(f"  close_mosaic = {args.close_mosaic}")
    print(f"  lr0          = {args.lr0}")
    print(f"  lrf          = {args.lrf}")
    print(f"  freeze       = {args.freeze}")
    print("  hsv_h/s/v    = 0.0 / 0.0 / 0.0")

    if args.fliplr > 0:
        print("[WARN] fliplr > 0 requires correct flip_idx in your dataset yaml.")
        print("[WARN] If left/right keypoints are sometimes swapped, rerun with --fliplr 0.0")

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        lr0=args.lr0,
        lrf=args.lrf,

        # grayscale-friendly aug
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        # pose-friendly conservative geometry aug
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
    )

    if args.freeze > 0:
        train_kwargs["freeze"] = args.freeze

    yolo.train(**train_kwargs)

    verify_first_conv(yolo)


if __name__ == "__main__":
    main()