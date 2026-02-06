import os
import sys
from pathlib import Path
import json

# Add third_party/moka to path so "from moka.xxx" works when running from project root
_project_root = Path(__file__).resolve().parent
_moka_dir = _project_root / "third_party" / "moka"
if str(_moka_dir) not in sys.path:
    sys.path.insert(0, str(_moka_dir))

from absl import app, flags
from easydict import EasyDict as edict
from PIL import Image
import matplotlib.pyplot as plt
import yaml

from openai import OpenAI

import numpy as np  # noqa: F401
import torch  # noqa: F401

from moka.gpt_utils import request_gpt  # noqa: F401
from moka.vision.segmentation import get_scene_object_bboxes, get_segmentation_masks
from moka.vision.keypoint import get_keypoints_from_segmentation  # noqa: F401
from moka.planners.planner import Planner  # noqa: F401
from moka.planners.visual_prompt_utils import *  # noqa: F401,F403


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "task_instruction",
    "Use the white ultrasound cleaner to clean the metal watch. "
    "The ultrasound cleaner has no lid and can be turned on by pressing the red button.",
    "Task description for MOKA.",
)
flags.DEFINE_bool(
    "show_plots",
    True,
    "Whether to show matplotlib figures (images, annotations).",
)


def load_config_and_paths():
    """Load MOKA config and resolve important paths relative to this repo."""
    project_root = Path(__file__).resolve().parent
    moka_dir = project_root / "third_party" / "moka"

    config_path = moka_dir / "config" / "moka.yaml"
    with open(config_path, "r") as fh:
        config = yaml.load(fh, Loader=yaml.SafeLoader)
    config = edict(config)

    # Make prompt_root_dir absolute if it is relative in the yaml
    if not os.path.isabs(config.prompt_root_dir):
        config.prompt_root_dir = str((moka_dir / config.prompt_root_dir).resolve())

    obs_image_path = moka_dir / "example" / "obs_image.jpg"

    return config, moka_dir, obs_image_path


def load_prompts(config):
    """Load prompts from files (adapted from demo.ipynb)."""
    prompts = dict()
    prompt_dir = os.path.join(config.prompt_root_dir, config.prompt_name)
    for filename in os.listdir(prompt_dir):
        path = os.path.join(prompt_dir, filename)
        if os.path.isfile(path) and path.endswith(".txt"):
            with open(path, "r") as f:
                value = f.read()
            key = filename[:-4]
            prompts[key] = value
    return prompts


def main(_):
    # Ensure API key is available (do NOT hardcode it here)
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError(
            "Please set the OPENAI_API_KEY environment variable before running "
            "`python test_demo.py`."
        )

    # Instantiate OpenAI client (used inside moka utilities)
    _client = OpenAI()

    config, moka_dir, obs_image_path = load_config_and_paths()
    prompts = load_prompts(config)

    # Load and show observation image
    obs_image = Image.open(obs_image_path).convert("RGB")
    obs_image = obs_image.resize([512, 512], Image.LANCZOS)

    if FLAGS.show_plots:
        plt.imshow(obs_image)
        plt.axis("off")
        plt.show()

    task_instruction = FLAGS.task_instruction
    print("Task:", task_instruction)

    # Plan using MOKA (same as notebook demo)
    plan = request_plan(
        task_instruction,
        obs_image,
        plan_with_obs_image=config.plan_with_obs_image,
        prompts=prompts,
        debug=True,
    )

    print("Raw plan:")
    try:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
    except TypeError:
        print(plan)

    # Collect all object names from the plan
    all_object_names = []
    for subtask in plan:
        if subtask.get("object_grasped"):
            if subtask["object_grasped"] not in all_object_names:
                all_object_names.append(subtask["object_grasped"])
        if subtask.get("object_unattached"):
            if subtask["object_unattached"] not in all_object_names:
                all_object_names.append(subtask["object_unattached"])

    print("Objects in scene:", all_object_names)

    # MOKA vision uses relative paths (./config, ./ckpts) - must run from moka dir
    orig_cwd = os.getcwd()
    try:
        os.chdir(moka_dir)
        # Get bounding boxes
        boxes, logits, phrases = get_scene_object_bboxes(
            obs_image,
            all_object_names,
            visualize=FLAGS.show_plots,
            logdir=None,
        )

        # Get segmentation masks
        segmasks = get_segmentation_masks(
            obs_image,
            all_object_names,
            boxes,
            logits,
            phrases,
            visualize=FLAGS.show_plots,
        )

        # Annotate visual prompts for the first subtask
        # subtask = plan[0]
        # candidate_keypoints = propose_candidate_keypoints(
        #     subtask,
        #     segmasks,
        #     num_samples=config.num_candidate_keypoints,
        # )

        # annotation_size = next(iter(segmasks.values()))["mask"].shape[:2][::-1]
        # obs_image_reshaped = obs_image.resize(annotation_size, Image.LANCZOS)

        # annotated_image = annotate_visual_prompts(
        #     obs_image_reshaped,
        #     candidate_keypoints,
        #     waypoint_grid_size=config.waypoint_grid_size,
        # )

        # if FLAGS.show_plots:
        #     plt.imshow(annotated_image)
        #     plt.axis("off")
        #     plt.show()

        # # Request motion based on annotated image
        # context, _, _ = request_motion(
        #     subtask,
        #     obs_image,
        #     annotated_image,
        #     candidate_keypoints,
        #     waypoint_grid_size=config.waypoint_grid_size,
        #     prompts=prompts,
        #     debug=True,
        # )

        # print("Motion context:")
        # print(context)
    finally:
        os.chdir(orig_cwd)


if __name__ == "__main__":
    app.run(main)



