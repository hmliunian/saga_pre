"""MOKA utilities for segmentation from image and task text."""

import os
from pathlib import Path

import yaml
from easydict import EasyDict as edict
from PIL import Image

# Resolve paths
_project_root = Path(__file__).resolve().parent.parent
_moka_dir = _project_root / "third_party" / "moka"


def _ensure_moka_in_path():
    """Add third_party/moka to sys.path for moka imports."""
    import sys

    if str(_moka_dir) not in sys.path:
        sys.path.insert(0, str(_moka_dir))


def _load_config_and_prompts():
    """Load MOKA config and prompts."""
    config_path = _moka_dir / "config" / "moka.yaml"
    with open(config_path, "r") as fh:
        config = yaml.load(fh, Loader=yaml.SafeLoader)
    config = edict(config)

    if not os.path.isabs(config.prompt_root_dir):
        config.prompt_root_dir = str((_moka_dir / config.prompt_root_dir).resolve())

    prompts = {}
    prompt_dir = os.path.join(config.prompt_root_dir, config.prompt_name)
    for filename in os.listdir(prompt_dir):
        path = os.path.join(prompt_dir, filename)
        if os.path.isfile(path) and path.endswith(".txt"):
            with open(path, "r") as f:
                value = f.read()
            key = filename[:-4]
            prompts[key] = value

    return config, prompts


def _extract_object_names_from_plan(plan):
    """Extract unique object names from MOKA plan (object_grasped, object_unattached)."""
    all_object_names = []
    for subtask in plan:
        if subtask.get("object_grasped") and subtask["object_grasped"] not in all_object_names:
            all_object_names.append(subtask["object_grasped"])
        if subtask.get("object_unattached") and subtask["object_unattached"] not in all_object_names:
            all_object_names.append(subtask["object_unattached"])
    return all_object_names


def get_segmasks_from_image_and_text(
    image: Image.Image,
    text: str,
    *,
    visualize: bool = False,
) -> dict:
    """
    Get segmentation masks for objects mentioned in the task text, detected in the image.

    Args:
        image: RGB PIL Image (not a file path).
        text: Task instruction / description (used to plan and extract object names).
        visualize: Whether to show matplotlib figures (default False).

    Returns:
        segmasks: Dict mapping object name to {'mask': ndarray, 'vertices': ndarray, 'bbox': list}.
        Same structure as moka.vision.segmentation.get_segmentation_masks output.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set (required for planning).
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError(
            "Please set the OPENAI_API_KEY environment variable before calling "
            "get_segmasks_from_image_and_text."
        )

    _ensure_moka_in_path()

    from openai import OpenAI
    from moka.gpt_utils import request_gpt
    from moka.planners.visual_prompt_utils import request_plan
    from moka.vision.segmentation import get_scene_object_bboxes, get_segmentation_masks

    _ = OpenAI()  # Used by moka utilities

    config, prompts = _load_config_and_prompts()

    plan = request_plan(
        text,
        image,
        plan_with_obs_image=config.plan_with_obs_image,
        prompts=prompts,
        debug=False,
    )

    all_object_names = _extract_object_names_from_plan(plan)
    if not all_object_names:
        return {}

    orig_cwd = os.getcwd()
    try:
        os.chdir(_moka_dir)
        boxes, logits, phrases = get_scene_object_bboxes(
            image,
            all_object_names,
            visualize=visualize,
            logdir=None,
        )
        segmasks = get_segmentation_masks(
            image,
            all_object_names,
            boxes,
            logits,
            phrases,
            visualize=visualize,
        )
        return segmasks
    finally:
        os.chdir(orig_cwd)


def _test_get_segmasks():
    """Test get_segmasks_from_image_and_text with example image and task text."""
    # example_image_path = _moka_dir / "example" / "obs_image.jpg"
    example_image_path = Path("/home/xuran-yao/code/june_serve/saga_pre/input/test.jpg")
    print(example_image_path)
    if not example_image_path.exists():
        raise FileNotFoundError(f"Example image not found: {example_image_path}")

    image = Image.open(example_image_path).convert("RGB")
    image = image.resize([512, 512], Image.LANCZOS)

    task_text = (
        # "Use the white ultrasound cleaner to clean the metal watch. "
        # "The ultrasound cleaner has no lid and can be turned on by pressing the red button."
        "Put the red toy on the pink plate."
    )

    print("Running get_segmasks_from_image_and_text...")
    segmasks = get_segmasks_from_image_and_text(image, task_text, visualize=False)

    print(f"Got {len(segmasks)} object(s): {list(segmasks.keys())}")
    for name, data in segmasks.items():
        mask = data["mask"]
        bbox = data["bbox"]
        print(f"  - {name}: mask shape={mask.shape}, bbox={bbox}")

    assert isinstance(segmasks, dict), "segmasks should be a dict"
    for name, data in segmasks.items():
        assert "mask" in data, f"{name} should have 'mask'"
        assert "vertices" in data, f"{name} should have 'vertices'"
        assert "bbox" in data, f"{name} should have 'bbox'"

    print("Test passed.")
    return segmasks


if __name__ == "__main__":
    _test_get_segmasks()
