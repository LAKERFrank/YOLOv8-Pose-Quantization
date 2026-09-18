#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a TRUE 1-channel grayscale YOLOv8n-Pose model from yolov8n-pose.pt.

- Dataset YAML must include: channels: 1
- Patch model first Conv2d: in_channels 3 -> 1 (weights averaged)
- Disable HSV aug (since grayscale)
"""

import argparse

import torch
import torch.nn as nn
from ultralytics import YOLO


def patch_first_conv_to_1ch(yolo: YOLO) -> None:
    """
    Replace the first nn.Conv2d(in_channels=3) with nn.Conv2d(in_channels=1),
    init weights by mean over RGB channels.
    """
    net = yolo.model  # torch.nn.Module inside ultralytics YOLO wrapper

    first_name, first_conv = None, None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3:
            first_name, first_conv = name, m
            break
    if first_conv is None:
        raise RuntimeError("Cannot find first Conv2d with in_channels=3 to patch.")

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
    )

    with torch.no_grad():
        # [out, 3, k, k] -> [out, 1, k, k]
        new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
        if first_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias)

    # assign new conv into the module tree
    if "." in first_name:
        parent_path, attr = first_name.rsplit(".", 1)
        parent = net.get_submodule(parent_path)
        setattr(parent, attr, new_conv)
    else:
        setattr(net, first_name, new_conv)

    # sanity print
    print(f"[OK] patched first conv: {first_name}: 3ch -> 1ch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset.yaml path (must contain channels: 1)")
    ap.add_argument("--model", default="yolov8n-pose.pt", help="pretrained weights")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="e.g. 0 / 0,1 / cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="train")
    args = ap.parse_args()

    # load pretrained
    yolo = YOLO(args.model)

    # patch model to true 1ch
    patch_first_conv_to_1ch(yolo)

    # train (disable HSV aug for grayscale)
    yolo.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,

        # grayscale-friendly aug
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )

    # verify exported model first conv
    net = yolo.model
    first_conv = None
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break
    print(f"[VERIFY] first conv in_channels = {first_conv.in_channels if first_conv else 'N/A'}")


if __name__ == "__main__":
    main()