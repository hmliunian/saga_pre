"""
Read and visualize PLY point clouds (xyz, rgb, label, heatmap) with Open3D.

- 可视化方式: heatmap_style
  - "brightness": 按 label 着色，heatmap 控明暗
  - "color": 按 heatmap 着色（0→蓝/冷色，1→红/热色），不同数值=不同颜色，非常明显
  - "blend": label 色与 heatmap 色混合，同时看到类别和强度
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _get_cmap(name: str):
    import matplotlib.pyplot as plt
    return plt.get_cmap(name)


def _label_colors_tab10() -> dict[int, tuple[float, float, float]]:
    """Label 0=灰；1,2,...= 区分度高的 RGB。"""
    cmap = _get_cmap("tab10")
    return {
        0: (0.12, 0.12, 0.12),
        1: cmap(0)[:3],
        2: cmap(1)[:3],
        3: cmap(2)[:3],
        4: cmap(3)[:3],
        5: cmap(4)[:3],
        6: cmap(5)[:3],
        7: cmap(6)[:3],
        8: cmap(7)[:3],
        9: cmap(8)[:3],
    }


@dataclass
class PointCloudLabelHeatmap:
    """点云：xyz, rgb, label, heatmap（与 main.py 输出格式一致）。"""

    xyz: Any  # (N, 3) float32
    rgb: Any  # (N, 3) uint8
    label: Any  # (N,) int32
    heatmap: Any  # (N,) float32 [0, 1]

    @property
    def n_points(self) -> int:
        return int(self.xyz.shape[0])

    def unique_labels(self):
        return sorted(set(int(x) for x in self.label))

    def colors_by_label_and_heatmap(
        self,
        *,
        heatmap_style: str = "color",
        heatmap_cmap: str = "coolwarm",
        heatmap_brightness: tuple[float, float] = (0.02, 1.0),
        heatmap_gamma: float = 0.45,
    ) -> np.ndarray:
        """
        计算可视化颜色 (N,3) float [0,1]。

        heatmap_style:
          - "color": 按 heatmap 着色，0=冷色(蓝)、1=热色(红)，不同数值=不同颜色，最明显
          - "brightness": 按 label 着色，heatmap 只控明暗
          - "blend": label 色与 heatmap 色混合，同时看类别和强度
        heatmap_cmap: 用于 "color"/"blend" 的色条，如 "coolwarm"(蓝→红)、"viridis"、"plasma"
        """
        t = np.clip(self.heatmap.astype(np.float64), 0.0, 1.0)
        t_vis = np.power(t, float(heatmap_gamma))
        cmap_heat = _get_cmap(heatmap_cmap)
        # (N,3) RGB from heatmap (matplotlib cmap accepts array)
        heat_rgb = np.asarray(cmap_heat(t_vis))[:, :3].astype(np.float64)

        if heatmap_style == "color":
            return np.clip(heat_rgb, 0.0, 1.0).astype(np.float32)

        label_to_rgb = _label_colors_tab10()
        labels = self.unique_labels()
        for lab in labels:
            if lab not in label_to_rgb and lab > 0:
                label_to_rgb[lab] = _get_cmap("tab10")((lab - 1) % 10)[:3]

        out = np.zeros((self.n_points, 3), dtype=np.float64)
        if heatmap_style == "brightness":
            lo, hi = heatmap_brightness
            for lab in labels:
                mask = self.label == lab
                base = np.array(label_to_rgb.get(lab, (0.2, 0.2, 0.2)), dtype=np.float64)
                b = lo + (hi - lo) * t_vis[mask]
                out[mask] = base * b[:, np.newaxis]
            return np.clip(out, 0.0, 1.0).astype(np.float32)

        # blend: (1 - t_vis) * label_color + t_vis * heat_rgb
        for lab in labels:
            mask = self.label == lab
            base = np.array(label_to_rgb.get(lab, (0.2, 0.2, 0.2)), dtype=np.float64)
            w = t_vis[mask][:, np.newaxis]
            out[mask] = (1 - w) * base + w * heat_rgb[mask]
        return np.clip(out, 0.0, 1.0).astype(np.float32)


def read_ply_xyz_rgb_label_heatmap(ply_path: str) -> PointCloudLabelHeatmap:
    """读取 ASCII PLY：x,y,z, red,green,blue, label, heatmap。"""
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)

    with open(ply_path, "r", encoding="utf-8") as f:
        if f.readline().strip() != "ply":
            raise ValueError("Not a PLY file")
        if f.readline().strip() != "format ascii 1.0":
            raise ValueError("Expected format ascii 1.0")

        n_vertex = None
        props = []
        while True:
            line = f.readline().strip()
            if line.startswith("element vertex "):
                n_vertex = int(line.split()[-1])
            elif line.startswith("property "):
                props.append(line.split()[2])
            elif line == "end_header":
                break

        if n_vertex is None:
            raise ValueError("Missing element vertex")

        name_to_idx = {p: i for i, p in enumerate(props)}
        for r in ["x", "y", "z", "red", "green", "blue", "label", "heatmap"]:
            if r not in name_to_idx:
                raise ValueError(f"Missing property {r}")

        col = lambda k: name_to_idx[k]
        rows = [f.readline().split() for _ in range(n_vertex)]

    data = np.array(rows, dtype=object)
    xyz = np.column_stack((
        data[:, col("x")].astype(np.float64),
        data[:, col("y")].astype(np.float64),
        data[:, col("z")].astype(np.float64),
    )).astype(np.float32)
    rgb = np.column_stack((
        data[:, col("red")].astype(np.int32),
        data[:, col("green")].astype(np.int32),
        data[:, col("blue")].astype(np.int32),
    )).astype(np.uint8)
    label = data[:, col("label")].astype(np.int32)
    heatmap = data[:, col("heatmap")].astype(np.float32)

    return PointCloudLabelHeatmap(xyz=xyz, rgb=rgb, label=label, heatmap=heatmap)


def visualize_ply_label_heatmap(
    ply_path: str,
    *,
    heatmap_style: str = "color",
    heatmap_cmap: str = "coolwarm",
    heatmap_brightness: tuple[float, float] = (0.02, 1.0),
    heatmap_gamma: float = 0.45,
    point_size: float = 1.5,
    show: bool = True,
    save: str | None = None,
) -> None:
    """
    Open3D 可视化。
    heatmap_style: "color"(按 heatmap 颜色，最明显) | "brightness"(明暗) | "blend"(混合)
    """
    import open3d as o3d

    pc = read_ply_xyz_rgb_label_heatmap(ply_path)
    colors = pc.colors_by_label_and_heatmap(
        heatmap_style=heatmap_style,
        heatmap_cmap=heatmap_cmap,
        heatmap_brightness=heatmap_brightness,
        heatmap_gamma=heatmap_gamma,
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pc.xyz, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Label + heatmap · {pc.n_points} pts", visible=show or bool(save))
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size

    if save and not show:
        for _ in range(10):
            vis.poll_events()
            vis.update_renderer()
        vis.capture_screen_image(save)
        vis.destroy_window()
        print(f"Saved: {save}")
    else:
        vis.run()
        vis.destroy_window()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Read & visualize PLY (label + heatmap) with Open3D")
    parser.add_argument("ply", nargs="?", default="output/pointcloud_label_heatmap.ply")
    parser.add_argument("-v", "--visualize", action="store_true", help="Open3D 交互可视化")
    parser.add_argument("--save-view", type=str, default=None, help="保存截图路径")
    parser.add_argument("--style", type=str, default="color", choices=("color", "brightness", "blend"),
                        help="color=按 heatmap 颜色(最明显) | brightness=明暗 | blend=混合")
    parser.add_argument("--heatmap-cmap", type=str, default="coolwarm",
                        help="color/blend 时色条: coolwarm(viridis/plasma 等)")
    parser.add_argument("--heatmap-min", type=float, default=0.02, help="brightness 时 heatmap=0 的亮度")
    parser.add_argument("--heatmap-max", type=float, default=1.0, help="brightness 时 heatmap=1 的亮度")
    parser.add_argument("--heatmap-gamma", type=float, default=0.45, help="gamma")
    args = parser.parse_args()

    pc = read_ply_xyz_rgb_label_heatmap(args.ply)

    if args.visualize or args.save_view:
        visualize_ply_label_heatmap(
            args.ply,
            heatmap_style=args.style,
            heatmap_cmap=args.heatmap_cmap,
            heatmap_brightness=(args.heatmap_min, args.heatmap_max),
            heatmap_gamma=args.heatmap_gamma,
            save=args.save_view,
            show=args.visualize,
        )

    print(f"Points: {pc.n_points}, labels: {pc.unique_labels()}")


if __name__ == "__main__":
    main()
