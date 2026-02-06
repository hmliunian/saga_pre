"""
Pipeline: image + text -> point cloud with xyz, rgb, label, heatmap.

Steps:
1. MOKA: get_segmasks_from_image_and_text -> segmasks (label names + masks)
2. CLIPSeg: clipseg_probs_full with labels as prompts -> heatmaps
3. Merge mask and heatmap per label -> pixel-level label_hw, heatmap_hw (label from 1; unlabeled = 0, heatmap = 0.0)
4. DA3: image_to_point_cloud_with_label_heatmap + write_ply_xyz_rgb_label_heatmap -> PLY
"""

from pathlib import Path

import numpy as np
from PIL import Image

from src.da3 import (
    image_to_point_cloud_with_label_heatmap,
    write_ply_xyz_rgb_label_heatmap,
)
from src.clipseg import clipseg_probs_full
from src.moka import get_segmasks_from_image_and_text


def _to_numpy(x):
    """Convert tensor to numpy if needed."""
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def run_pipeline(
    image_path: str,
    text: str,
    output_ply_path: str,
    *,
    image_size: int = 512,
    device_clipseg: str = "cuda",
    device_da3: str = None,
    max_points: int = 500_000,
) -> None:
    """
    Run full pipeline: image + text -> PLY with xyz, rgb, label, heatmap.

    - Labels from MOKA start at 1; pixels with no mask get label=0, heatmap=0.0.
    """
    # Load and optionally resize image for consistent resolution across MOKA / CLIPSeg / DA3
    image = Image.open(image_path).convert("RGB")
    # if image_size and (image.size[0] != image_size or image.size[1] != image_size):
    #     image = image.resize((image_size, image_size), Image.LANCZOS)
    h, w = image.size[1], image.size[0]

    # Step 1: MOKA -> segmasks (label names + masks)
    segmasks = get_segmasks_from_image_and_text(image, text, visualize=False)
    labels_list = list(segmasks.keys())  # ordered list of object names

    if len(labels_list) == 0:
        # No objects: full image has label=0, heatmap=0.0
        label_hw = np.zeros((h, w), dtype=np.int32)
        heatmap_hw = np.zeros((h, w), dtype=np.float32)
    else:
        # Step 2: CLIPSeg with prompts = label names -> (N, 1, H, W)
        probs_full = clipseg_probs_full(
            image,
            labels_list,
            device=device_clipseg,
        )
        # probs_full: (N, 1, H_orig, W_orig)
        probs_full = _to_numpy(probs_full)
        if probs_full.shape[2] != h or probs_full.shape[3] != w:
            # Resize to image size if CLIPSeg returned different size
            from src.da3 import _resize_hw
            probs_resized = np.zeros((len(labels_list), 1, h, w), dtype=np.float32)
            for i in range(len(labels_list)):
                ch = probs_full[i, 0]
                probs_resized[i, 0] = _resize_hw(ch, (h, w), mode="linear")
            probs_full = probs_resized

        # Step 3: Pixel-level label_hw and heatmap_hw (label 1..K, unlabeled 0)
        label_hw = np.zeros((h, w), dtype=np.int32)
        heatmap_hw = np.zeros((h, w), dtype=np.float32)

        for i, name in enumerate(labels_list):
            mask_i = np.asarray(segmasks[name]["mask"])
            if mask_i.shape[0] != h or mask_i.shape[1] != w:
                from src.da3 import _resize_hw
                mask_i = _resize_hw(
                    mask_i.astype(np.float32), (h, w), mode="nearest"
                ).astype(bool)
            else:
                mask_i = mask_i.astype(bool)

            heat_i = probs_full[i, 0]  # (H, W)
            # Assign where this mask is True and not yet assigned (first object wins)
            where = mask_i & (label_hw == 0)
            label_hw[where] = i + 1
            heatmap_hw[where] = np.clip(heat_i[where].astype(np.float32), 0.0, 1.0)

    # Step 4: DA3 -> point cloud with label/heatmap, then write PLY
    pc = image_to_point_cloud_with_label_heatmap(
        image=image,
        label_hw=label_hw,
        heatmap_hw=heatmap_hw,
        model=None,
        model_dir=None,
        device=device_da3,
        max_points=max_points,
        conf_percentile=None,
        exclude_sky=False,
    )

    # 对比输入图像像素数与生成点云点数，做判断
    image_pixels = h * w
    da3_depth_h, da3_depth_w = pc.depth_hw.shape[0], pc.depth_hw.shape[1]
    da3_pixels = da3_depth_h * da3_depth_w
    output_points = pc.xyz_world.shape[0]

    print(
        f"Pixel/point counts: input image {h}x{w} = {image_pixels}, "
        f"DA3 depth {da3_depth_h}x{da3_depth_w} = {da3_pixels}, "
        f"output points = {output_points}"
    )
    if output_points == 0:
        raise RuntimeError(
            "生成点云为空。请检查深度/天空/置信度过滤或输入图像。"
        )
    ratio_vs_da3 = output_points / da3_pixels
    if ratio_vs_da3 < 0.05:
        print(
            f"Warning: 仅 {ratio_vs_da3*100:.1f}% 的 DA3 像素成为有效点 "
            f"(深度无效/天空/置信度过滤或 max_points={max_points} 下采样)。"
        )
    if output_points < image_pixels * 0.01:
        print(
            f"Warning: 输出点数 ({output_points}) 远小于输入图像像素数 ({image_pixels})，"
            "部分点因深度过滤或下采样丢失。"
        )

    Path(output_ply_path).parent.mkdir(parents=True, exist_ok=True)
    write_ply_xyz_rgb_label_heatmap(
        output_ply_path,
        xyz_world=pc.xyz_world,
        rgb_uint8=pc.rgb_uint8,
        label_int32=pc.label_int32,
        heatmap_float32=pc.heatmap_float32,
    )
    print(f"Wrote {output_ply_path} (points: {output_points}, labels: 0..{int(np.max(pc.label_int32))})")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Image + text -> point cloud (xyz, rgb, label, heatmap)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="input/test.jpg",
        help="Input image path",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Put the red toy on the pink plate.",
        help="Task / instruction text",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/pointcloud_label_heatmap.ply",
        help="Output PLY path",
    )
    parser.add_argument("--image-size", type=int, default=512, help="Resize image to this (H,W)")
    parser.add_argument("--device-clipseg", type=str, default="cuda", help="Device for CLIPSeg")
    parser.add_argument("--device-da3", type=str, default=None, help="Device for DA3 (default: auto)")
    parser.add_argument("--max-points", type=int, default=500_000, help="Max points in point cloud")
    args = parser.parse_args()

    run_pipeline(
        args.image,
        args.text,
        args.output,
        image_size=args.image_size,
        device_clipseg=args.device_clipseg,
        device_da3=args.device_da3,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()
