"""
Depth Anything 3 helpers for generating point clouds with per-point attributes.

This module is intentionally self-contained (no Open3D dependency).
It follows the logic in `third_party/Depth-Anything-3/test_da3.py`, and extends it
to map per-pixel `label` (int) and `heatmap` (float in [0,1]) onto the point cloud.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np


def _ensure_depth_anything3_importable() -> None:
    """
    Make `depth_anything_3` importable without requiring an editable install.

    We add `<repo>/third_party/Depth-Anything-3/src` to sys.path if needed.
    """

    try:
        import depth_anything_3  # noqa: F401

        return
    except Exception:
        pass

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    da3_src = os.path.join(repo_root, "third_party", "Depth-Anything-3", "src")
    if os.path.isdir(da3_src) and da3_src not in sys.path:
        sys.path.insert(0, da3_src)


def _as_w2c_4x4(extrinsic: Optional[np.ndarray]) -> np.ndarray:
    """
    DepthAnything3 `Prediction.extrinsics` could be:
    - None
    - (3, 4) w2c
    - (4, 4) w2c
    """

    if extrinsic is None:
        return np.eye(4, dtype=np.float32)
    ex = np.asarray(extrinsic)
    if ex.shape == (4, 4):
        return ex.astype(np.float32)
    if ex.shape == (3, 4):
        ex4 = np.eye(4, dtype=np.float32)
        ex4[:3, :] = ex.astype(np.float32)
        return ex4
    raise ValueError(f"Unsupported extrinsic shape: {ex.shape}")


def _as_ixt_3x3(
    intrinsic: Optional[np.ndarray],
    hw: Tuple[int, int],
    fov_deg: float = 60.0,
) -> np.ndarray:
    """
    If model doesn't output intrinsics, create a reasonable pinhole K.
    """

    if intrinsic is not None:
        K = np.asarray(intrinsic).astype(np.float32)
        if K.shape == (3, 3):
            return K
        raise ValueError(f"Unsupported intrinsic shape: {K.shape}")

    h, w = hw
    fov = np.deg2rad(float(fov_deg))
    fx = 0.5 * w / np.tan(0.5 * fov)
    fy = fx
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _resize_hw(
    arr_hw: np.ndarray,
    out_hw: Tuple[int, int],
    *,
    mode: str,
) -> np.ndarray:
    """
    Resize a (H,W) array to `out_hw=(H,W)`.

    mode:
      - "nearest" for discrete labels
      - "linear" for heatmaps
    """

    arr = np.asarray(arr_hw)
    if arr.ndim != 2:
        raise ValueError(f"Expected (H,W), got {arr.shape}")
    in_h, in_w = int(arr.shape[0]), int(arr.shape[1])
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if (in_h, in_w) == (out_h, out_w):
        return arr

    # Prefer OpenCV if available, else fall back to PIL.
    try:
        import cv2  # type: ignore

        interp = cv2.INTER_NEAREST if mode == "nearest" else cv2.INTER_LINEAR
        # cv2.resize expects (W,H)
        return cv2.resize(arr, (out_w, out_h), interpolation=interp)
    except Exception:
        pass

    try:
        from PIL import Image  # type: ignore

        pil_mode = Image.NEAREST if mode == "nearest" else Image.BILINEAR
        im = Image.fromarray(arr)
        im = im.resize((out_w, out_h), resample=pil_mode)
        return np.asarray(im)
    except Exception as e:
        raise RuntimeError(
            "Failed to resize label/heatmap. Install either `opencv-python` or `Pillow`."
        ) from e


@dataclass(frozen=True)
class PointCloudWithAttrs:
    """
    A lightweight point cloud container (no external deps).
    """

    xyz_world: np.ndarray  # (N,3) float32
    rgb_uint8: np.ndarray  # (N,3) uint8 (RGB)
    label_int32: np.ndarray  # (N,) int32
    heatmap_float32: np.ndarray  # (N,) float32 in [0,1]
    intrinsics_3x3: np.ndarray  # (3,3) float32
    extrinsics_w2c_4x4: np.ndarray  # (4,4) float32
    depth_hw: np.ndarray  # (H,W) float32
    processed_rgb_hw3: np.ndarray  # (H,W,3) uint8


def load_da3_model(
    *,
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
):
    """
    Load DepthAnything3 model.

    - model_dir can be a HF repo id (e.g. "depth-anything/DA3NESTED-GIANT-LARGE")
      or a local directory.
    - device: "cuda" / "cpu". If None, choose automatically.
    """

    _ensure_depth_anything3_importable()
    import torch
    from depth_anything_3.api import DepthAnything3

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_dir is None:
        # Prefer the repo-bundled local checkpoint if present.
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(here, ".."))
        candidate = os.path.join(
            repo_root, "third_party", "Depth-Anything-3", "hf_models", "DA3NESTED-GIANT-LARGE"
        )
        model_dir = candidate if os.path.exists(candidate) else "depth-anything/DA3NESTED-GIANT-LARGE"
    else:
        # Resolve relative local paths from repo root if they exist.
        if not os.path.isabs(model_dir):
            here = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.abspath(os.path.join(here, ".."))
            candidate = os.path.join(repo_root, model_dir)
            if os.path.exists(candidate):
                model_dir = candidate

    dev = torch.device(device)
    model = DepthAnything3.from_pretrained(model_dir).to(dev)
    return model


def image_to_point_cloud_with_label_heatmap(
    *,
    image: Union[str, np.ndarray, Any],
    label_hw: Optional[np.ndarray] = None,
    heatmap_hw: Optional[np.ndarray] = None,
    model: Optional[Any] = None,
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    fov_deg_if_missing_ixt: float = 60.0,
    conf_percentile: Optional[float] = 40.0,
    max_points: int = 1_000_000,
    rng_seed: int = 42,
    exclude_sky: bool = True,
) -> PointCloudWithAttrs:
    """
    Single image -> point cloud with per-point attributes (xyz/rgb/label/heatmap).

    Inputs:
    - image: file path, numpy RGB/BGR image, or PIL.Image
    - label_hw: per-pixel int labels aligned with `image` pixels (H,W). Optional.
    - heatmap_hw: per-pixel float heatmap in [0,1] aligned with `image` pixels (H,W). Optional.

    Mapping rules:
    - We run DA3 to get depth at its processed resolution (H',W').
    - If label/heatmap are provided and their shape != (H',W'), we resize:
        - label: nearest
        - heatmap: linear
    - If label/heatmap are not provided, per-point label/heatmap are set to 0.
    - If points are filtered/dowsampled, the attributes stay 1:1 with remaining points.
    """

    if model is None:
        model = load_da3_model(model_dir=model_dir, device=device)

    # DA3 inference
    prediction = model.inference(
        image=[image],
        process_res=int(process_res),
        process_res_method=str(process_res_method),
        export_dir=None,
        export_format="mini_npz",
    )

    depth = np.asarray(prediction.depth[0]).astype(np.float32)  # (H,W)
    rgb = prediction.processed_images[0] if prediction.processed_images is not None else None
    if rgb is None:
        raise RuntimeError("prediction.processed_images is None; expected image for point colors")
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"processed_images[0] must be (H,W,3), got {rgb.shape}")

    intr = prediction.intrinsics[0] if prediction.intrinsics is not None else None
    extr = prediction.extrinsics[0] if prediction.extrinsics is not None else None
    K = _as_ixt_3x3(intr, depth.shape, fov_deg=float(fov_deg_if_missing_ixt))
    w2c = _as_w2c_4x4(extr)

    conf = prediction.conf[0] if prediction.conf is not None else None
    sky = prediction.sky[0] if prediction.sky is not None else None

    h, w = int(depth.shape[0]), int(depth.shape[1])

    # Prepare label / heatmap at depth resolution
    if label_hw is None:
        label_rs = np.zeros((h, w), dtype=np.int32)
    else:
        lab = np.asarray(label_hw)
        if lab.ndim != 2:
            raise ValueError(f"label_hw must be (H,W), got {lab.shape}")
        label_rs = _resize_hw(lab, (h, w), mode="nearest").astype(np.int32, copy=False)

    if heatmap_hw is None:
        heat_rs = np.zeros((h, w), dtype=np.float32)
    else:
        hm = np.asarray(heatmap_hw)
        if hm.ndim != 2:
            raise ValueError(f"heatmap_hw must be (H,W), got {hm.shape}")
        heat_rs = _resize_hw(hm.astype(np.float32, copy=False), (h, w), mode="linear").astype(
            np.float32, copy=False
        )
        # best-effort clamp to [0,1]
        heat_rs = np.clip(heat_rs, 0.0, 1.0)

    # Pixel grid
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    valid = np.isfinite(depth) & (depth > 0)

    if exclude_sky and sky is not None:
        sky_hw = np.asarray(sky)
        if sky_hw.shape == depth.shape:
            valid &= ~(sky_hw > 0.5)

    if conf is not None and conf_percentile is not None:
        conf_hw = np.asarray(conf).astype(np.float32)
        if conf_hw.shape == depth.shape:
            conf_valid = conf_hw[valid]
            if conf_valid.size > 0:
                thr = np.percentile(conf_valid, float(conf_percentile))
                valid &= conf_hw >= thr

    if not np.any(valid):
        zeros3 = np.zeros((0, 3), dtype=np.float32)
        zerosrgb = np.zeros((0, 3), dtype=np.uint8)
        return PointCloudWithAttrs(
            xyz_world=zeros3,
            rgb_uint8=zerosrgb,
            label_int32=np.zeros((0,), dtype=np.int32),
            heatmap_float32=np.zeros((0,), dtype=np.float32),
            intrinsics_3x3=K.astype(np.float32),
            extrinsics_w2c_4x4=w2c.astype(np.float32),
            depth_hw=depth.astype(np.float32),
            processed_rgb_hw3=rgb.astype(np.uint8),
        )

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    z = depth[valid]
    x = (u[valid] - cx) / fx * z
    y = (v[valid] - cy) / fy * z
    points_cam = np.stack([x, y, z], axis=1).astype(np.float32)  # (N,3)

    # cam -> world
    c2w = np.linalg.inv(w2c).astype(np.float32)
    xyz_world = (c2w[:3, :3] @ points_cam.T).T + c2w[:3, 3]

    rgb_uint8 = rgb[valid].astype(np.uint8)
    label_vec = label_rs[valid].astype(np.int32, copy=False)
    heat_vec = heat_rs[valid].astype(np.float32, copy=False)

    if xyz_world.shape[0] > int(max_points):
        rng = np.random.default_rng(int(rng_seed))
        idx = rng.choice(xyz_world.shape[0], size=int(max_points), replace=False)
        xyz_world = xyz_world[idx]
        rgb_uint8 = rgb_uint8[idx]
        label_vec = label_vec[idx]
        heat_vec = heat_vec[idx]

    return PointCloudWithAttrs(
        xyz_world=xyz_world.astype(np.float32, copy=False),
        rgb_uint8=rgb_uint8.astype(np.uint8, copy=False),
        label_int32=label_vec.astype(np.int32, copy=False),
        heatmap_float32=heat_vec.astype(np.float32, copy=False),
        intrinsics_3x3=K.astype(np.float32, copy=False),
        extrinsics_w2c_4x4=w2c.astype(np.float32, copy=False),
        depth_hw=depth.astype(np.float32, copy=False),
        processed_rgb_hw3=rgb.astype(np.uint8, copy=False),
    )


def write_ply_xyz_rgb_label_heatmap(
    ply_path: str,
    *,
    xyz_world: np.ndarray,
    rgb_uint8: np.ndarray,
    label_int32: Optional[np.ndarray] = None,
    heatmap_float32: Optional[np.ndarray] = None,
) -> None:
    """
    Write an ASCII PLY point cloud with properties:
      x y z (float), red green blue (uchar), label (int), heatmap (float)

    If label/heatmap are None, they will be written as 0 for all points.
    """

    pts = np.asarray(xyz_world)
    cols = np.asarray(rgb_uint8)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"xyz_world must be (N,3), got {pts.shape}")
    if cols.ndim != 2 or cols.shape[1] != 3:
        raise ValueError(f"rgb_uint8 must be (N,3), got {cols.shape}")
    if pts.shape[0] != cols.shape[0]:
        raise ValueError(f"N mismatch: xyz_world={pts.shape}, rgb_uint8={cols.shape}")

    n = int(pts.shape[0])
    if label_int32 is None:
        lab = np.zeros((n,), dtype=np.int32)
    else:
        lab = np.asarray(label_int32).astype(np.int32, copy=False)
        if lab.shape != (n,):
            raise ValueError(f"label_int32 must be (N,), got {lab.shape}")

    if heatmap_float32 is None:
        hm = np.zeros((n,), dtype=np.float32)
    else:
        hm = np.asarray(heatmap_float32).astype(np.float32, copy=False)
        if hm.shape != (n,):
            raise ValueError(f"heatmap_float32 must be (N,), got {hm.shape}")
        hm = np.clip(hm, 0.0, 1.0)

    os.makedirs(os.path.dirname(os.path.abspath(ply_path)) or ".", exist_ok=True)
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int label\n")
        f.write("property float heatmap\n")
        f.write("end_header\n")

        for p, c, li, hi in zip(pts, cols, lab, hm):
            f.write(
                f"{float(p[0])} {float(p[1])} {float(p[2])} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} "
                f"{int(li)} {float(hi)}\n"
            )

def _read_image_hw(image_path: str) -> Tuple[int, int]:
    """
    Read image and return (H, W) for generating random label/heatmap.
    Prefer cv2 if available, else PIL.
    """
    try:
        import cv2  # type: ignore

        im = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"cv2.imread failed: {image_path}")
        h, w = im.shape[:2]
        return int(h), int(w)
    except Exception:
        pass

    try:
        from PIL import Image  # type: ignore

        im = Image.open(image_path)
        w, h = im.size
        return int(h), int(w)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read image for shape. Install `opencv-python` or `Pillow`. Path={image_path}"
        ) from e


def test_one_image_random_label_heatmap() -> None:
    # ========= 1) set your image path here =========
    image_path = "/home/xuran-yao/code/june_serve/saga_pre/third_party/Depth-Anything-3/assets/examples/003.png"  # <-- 改成你的一张图路径

    # ========= 2) generate random label + heatmap aligned to original image =========
    H0, W0 = _read_image_hw(image_path)
    rng = np.random.default_rng(123)

    num_classes = 8
    label_hw = rng.integers(low=0, high=num_classes, size=(H0, W0), dtype=np.int32)
    heatmap_hw = rng.random(size=(H0, W0), dtype=np.float32)  # [0,1)

    print("[TEST] image_path:", image_path)
    print("[TEST] original HW:", (H0, W0))
    print("[TEST] label_hw:", label_hw.shape, label_hw.dtype, "min/max:", int(label_hw.min()), int(label_hw.max()))
    print("[TEST] heatmap_hw:", heatmap_hw.shape, heatmap_hw.dtype, "min/max:", float(heatmap_hw.min()), float(heatmap_hw.max()))

    # ========= 3) run DA3 -> point cloud with per-point attrs =========
    pc = image_to_point_cloud_with_label_heatmap(
        image=image_path,
        label_hw=label_hw,
        heatmap_hw=heatmap_hw,
        model=None,                 # let helper load model
        model_dir=None,             # default: local checkpoint if present else HF repo
        device=None,                # auto
        process_res=504,
        process_res_method="upper_bound_resize",
        conf_percentile=40.0,
        max_points=200_000,          # keep it smaller for a quick test
        exclude_sky=True,
    )

    print("[TEST] xyz_world:", pc.xyz_world.shape, pc.xyz_world.dtype)
    print("[TEST] rgb_uint8:", pc.rgb_uint8.shape, pc.rgb_uint8.dtype, "min/max:", int(pc.rgb_uint8.min()), int(pc.rgb_uint8.max()))
    print("[TEST] label_int32:", pc.label_int32.shape, pc.label_int32.dtype, "unique:", int(np.unique(pc.label_int32).size))
    print("[TEST] heatmap_float32:", pc.heatmap_float32.shape, pc.heatmap_float32.dtype,
          "min/max:", float(pc.heatmap_float32.min(initial=0.0)), float(pc.heatmap_float32.max(initial=0.0)))

    # ========= 4) write ply =========
    out_ply = "output/test_da3_random_attrs.ply"
    write_ply_xyz_rgb_label_heatmap(
        out_ply,
        xyz_world=pc.xyz_world,
        rgb_uint8=pc.rgb_uint8,
        label_int32=pc.label_int32,
        heatmap_float32=pc.heatmap_float32,
    )
    print("[TEST] wrote:", out_ply)


if __name__ == "__main__":
    test_one_image_random_label_heatmap()