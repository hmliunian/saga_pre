"""
CLIPSeg wrapper.

This module ports logic from `third_party/clipseg/test.py` into a reusable function.

Function contract (per request):
- input: one image + multiple text prompts (e.g. ["metal watch", "ultrasound cleaner", "red button"])
- output: `probs_full` from `unletterbox_to_orig(probs_352, meta)`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, TypeAlias

from PIL import Image


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CLIPSEG_DIR = _PROJECT_ROOT / "third_party" / "clipseg"

# Optional typing without requiring torch at import time.
try:  # pragma: no cover
    import torch as _torch  # type: ignore

    TorchTensor: TypeAlias = _torch.Tensor
    TorchDevice: TypeAlias = _torch.device
except ModuleNotFoundError:  # pragma: no cover
    TorchTensor: TypeAlias = Any
    TorchDevice: TypeAlias = Any


def _ensure_clipseg_in_path() -> None:
    """Add `third_party/clipseg` to sys.path so `from models.clipseg import ...` works."""
    import sys

    if str(_CLIPSEG_DIR) not in sys.path:
        sys.path.insert(0, str(_CLIPSEG_DIR))


@dataclass(frozen=True)
class LetterboxMeta:
    orig_w: int
    orig_h: int
    target: int
    scale: float
    new_w: int
    new_h: int
    pad_left: int
    pad_top: int


def letterbox_pad(pil_img: Image.Image, target: int = 352, fill: tuple[int, int, int] = (0, 0, 0)) -> tuple[Image.Image, LetterboxMeta]:
    """
    Keep aspect ratio, resize to fit within target x target, then pad to square.
    Returns:
      canvas (target,target), meta for mapping back.
    """
    w, h = pil_img.size
    scale = min(target / w, target / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target, target), fill)

    pad_left = (target - new_w) // 2
    pad_top = (target - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))

    meta = LetterboxMeta(
        orig_w=w,
        orig_h=h,
        target=target,
        scale=scale,
        new_w=new_w,
        new_h=new_h,
        pad_left=pad_left,
        pad_top=pad_top,
    )
    return canvas, meta


def unletterbox_to_orig(probs_352: TorchTensor, meta: LetterboxMeta) -> TorchTensor:
    """
    probs_352: (N,1,target,target)
    1) crop away padding to (N,1,new_h,new_w)
    2) interpolate back to (N,1,orig_h,orig_w)
    """
    torch, F, _ = _require_torch()

    pad_left, pad_top = meta.pad_left, meta.pad_top
    new_w, new_h = meta.new_w, meta.new_h
    orig_w, orig_h = meta.orig_w, meta.orig_h

    # crop content region
    probs_content = probs_352[:, :, pad_top : pad_top + new_h, pad_left : pad_left + new_w]  # (N,1,new_h,new_w)

    # back to original size
    probs_full = F.interpolate(
        probs_content,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )
    return probs_full  # (N,1,orig_h,orig_w)


def _require_torch() -> tuple[Any, Any, Any]:
    """
    Import torch/torchvision lazily so this module can be imported even when
    the current Python environment doesn't have those deps installed.
    """
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
        from torchvision import transforms  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency. Please run with a Python environment that has "
            "`torch` and `torchvision` installed.\n\n"
            "If you're using the vendored moka env from this repo, try:\n"
            f'  "{_PROJECT_ROOT / "third_party" / "moka" / "mk" / "bin" / "python"}" -m src.clipseg\n'
        ) from e
    return torch, F, transforms


def _default_transform():
    _, _, transforms = _require_torch()
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _load_model(
    *,
    weights_path: str | Path,
    device: TorchDevice,
    version: str = "ViT-B/16",
    reduce_dim: int = 64,
) -> Any:
    """
    Load CLIPSeg model with weights. Cached by (weights_path, device).
    """
    torch, _, _ = _require_torch()
    _ensure_clipseg_in_path()
    from models.clipseg import CLIPDensePredT  # type: ignore

    weights_path = Path(weights_path)
    cache_key = (str(weights_path.resolve()), str(device))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model = CLIPDensePredT(version=version, reduce_dim=reduce_dim)
    model.eval()
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)

    _MODEL_CACHE[cache_key] = model
    return model


def clipseg_probs_full(
    image: Image.Image,
    prompts: Sequence[str],
    *,
    target: int = 352,
    weights_path: str | Path | None = None,
    device: str | TorchDevice = "cpu",
    version: str = "ViT-B/16",
    reduce_dim: int = 64,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> TorchTensor:
    """
    Args:
        image: One RGB PIL Image.
        prompts: Text prompts list.
        target: Letterbox target size (square). Should be divisible by 16 for ViT-B/16.
        weights_path: Path to `rd64-uni.pth`. Defaults to `third_party/clipseg/weights/rd64-uni.pth`.
        device: "cpu" or "cuda".
        version: CLIP backbone version.
        reduce_dim: CLIPSeg reduce dim.
        fill: Letterbox pad color.

    Returns:
        probs_full: torch.Tensor with shape (N, 1, H_orig, W_orig)
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"`image` must be a PIL.Image.Image, got {type(image)}")
    if image.mode != "RGB":
        image = image.convert("RGB")
    if not isinstance(prompts, (list, tuple)) or len(prompts) == 0:
        raise ValueError("`prompts` must be a non-empty list/tuple of strings.")

    # Force square input (important): CLIPSeg assumes square patch grid when rescaling pos embedding.
    # Non-square inputs can cause: seq_len-1 not a perfect square -> pos-emb mismatch.
    torch, _, _ = _require_torch()

    lb_img, meta = letterbox_pad(image, target=target, fill=fill)

    img_t = _default_transform()(lb_img).unsqueeze(0)  # [1,3,target,target]

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    img_t = img_t.to(dev)

    if weights_path is None:
        weights_path = _CLIPSEG_DIR / "weights" / "rd64-uni.pth"

    model = _load_model(weights_path=weights_path, device=dev, version=version, reduce_dim=reduce_dim)

    with torch.no_grad():
        preds = model(img_t.repeat(len(prompts), 1, 1, 1), list(prompts))[0]  # (N,1,target,target)
        probs_352 = torch.sigmoid(preds)

    probs_full = unletterbox_to_orig(probs_352, meta)  # (N,1,H0,W0)
    return probs_full


def _main(argv: Iterable[str] | None = None) -> int:
    """
    Verification entrypoint.

    Example:
      python -m src.clipseg --image third_party/moka/example/obs_image.jpg --prompts "metal watch" "ultrasound cleaner" "red button"
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run CLIPSeg on an image with multiple prompts.")
    parser.add_argument(
        "--image",
        type=str,
        default=str(_PROJECT_ROOT / "third_party" / "moka" / "example" / "obs_image.jpg"),
        help="Path to an input image.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["metal watch", "ultrasound cleaner", "red button"],
        help="One or more text prompts.",
    )
    parser.add_argument("--device", type=str, default="cuda", help='Device, e.g. "cpu" or "cuda".')
    parser.add_argument("--target", type=int, default=352, help="Letterbox target square size.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(_CLIPSEG_DIR / "weights" / "rd64-uni.pth"),
        help="Path to CLIPSeg weights file.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    img = Image.open(args.image).convert("RGB")
    probs_full = clipseg_probs_full(
        img,
        args.prompts,
        target=args.target,
        weights_path=args.weights,
        device=args.device,
    )

    w0, h0 = img.size
    print("image.size (W,H):", (w0, h0))
    print("num prompts:", len(args.prompts))
    print("probs_full.shape:", tuple(probs_full.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
