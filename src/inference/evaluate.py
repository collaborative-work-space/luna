# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Modified, aR, 2024.
 
import os
import sys
import argparse
import numpy as np
import torch

from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from dataset.preprocessing.preprocess_scannet import calculate_iou_folders, calculate_panoptic_quality_folders
from pathlib import Path
from util.metrics import psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision.transforms.functional import to_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='metrics')
    parser.add_argument('--root_path', required=False, default='data/scannet/scene0423_02')
    parser.add_argument('--exp_path', required=False, default='runs/scene0423_02_test_01170151_PanopLi_scannet042302_electrical-forest')
    args = parser.parse_args()

    print('calculating metrics for ours')
    image_dim = (512, 512)
    iou = calculate_iou_folders(Path(args.exp_path, "pred_semantics"), Path(args.root_path) / "rs_semantics", image_dim)
    pq, rq, sq = calculate_panoptic_quality_folders(Path(args.exp_path, "pred_semantics"), Path(args.exp_path, "pred_surrogateid"), Path(args.root_path) / "rs_semantics", Path(args.root_path) / "rs_instance", image_dim)
    print(f'[dataset] iou, pq, sq, rq: {iou:.3f}, {pq:.3f}, {sq:.3f}, {rq:.3f}')

    print("Calculating PSNR and SSIM...")

    pred_rgb_dir = Path(args.exp_path) / "pred_rgb_raw"
    gt_rgb_dir = Path(args.root_path) / "color"

    psnr_vals, ssim_vals = [], []

    for pred_path in sorted(pred_rgb_dir.glob("*.npy")):
        name = pred_path.stem + ".png"  # e.g., 0_0004.png
        gt_path = gt_rgb_dir / name
        if not gt_path.exists():
            print(f"Skipping missing GT image: {name}")
            continue

        # Load prediction and ground truth
        pred_np = np.load(pred_path)  # [H, W, 3], float32, [0, 1]
        gt_np = np.array(Image.open(gt_path).convert("RGB")).astype(np.float32) / 255.0

        # Resize GT if necessary to match pred
        if pred_np.shape != gt_np.shape:
            gt_np = np.array(Image.fromarray((gt_np * 255).astype(np.uint8)).resize(
                (pred_np.shape[1], pred_np.shape[0]), Image.BILINEAR)).astype(np.float32) / 255.0

        # Convert to tensor
        pred_tensor = torch.tensor(pred_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        gt_tensor = torch.tensor(gt_np).permute(2, 0, 1).unsqueeze(0)

        # PSNR from util.metrics
        psnr_val = psnr(pred_tensor, gt_tensor)
        psnr_vals.append(psnr_val.item())

        # SSIM from skimage
        ssim_val = compare_ssim(pred_np, gt_np, channel_axis=2)
        ssim_vals.append(ssim_val)

    # Final metrics
    avg_psnr = np.mean(psnr_vals)
    avg_ssim = np.mean(ssim_vals)

    print(f'[dataset] PSNR: {avg_psnr:.3f}, SSIM: {avg_ssim:.3f}')
