# Adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_helpers.py#L43

from pathlib import Path
from typing import OrderedDict

import torch
from tqdm import tqdm

STEPS_TO_AVER_HITRATE = [20, 25, 31]
STEPS_TO_AVER_MAP = [20, 25, 28]

ALL_CHECKPOINTS_PATH = Path(__file__).parent / "checkpoints"
AVER_CHECKPOINTS_PATH = Path(__file__).parent / "aver_checkpoints"


def main():
    hitrate_checkpoints_paths = get_checkpoints_paths(
        ALL_CHECKPOINTS_PATH, STEPS_TO_AVER_HITRATE
    )
    mAP_checkpoints_paths = get_checkpoints_paths(
        ALL_CHECKPOINTS_PATH, STEPS_TO_AVER_MAP
    )
    checkpoints_paths = hitrate_checkpoints_paths + mAP_checkpoints_paths

    avg_state_dict, avg_counts = dict(), dict()
    pbar = tqdm(checkpoints_paths, desc=f"Checkpoints collecting")
    for checkpoint_path in pbar:
        checkpoint: OrderedDict = torch.load(checkpoint_path, map_location="cpu")
        state_dict: OrderedDict = checkpoint["generator"]
        for k, v in state_dict.items():
            if k not in avg_state_dict:
                avg_state_dict[k] = v.clone().to(dtype=torch.float64)
                avg_counts[k] = 1
            else:
                avg_state_dict[k] += v.to(dtype=torch.float64)
                avg_counts[k] += 1

    for k, v in avg_state_dict.items():
        v.div_(avg_counts[k])

    # float32 overflow seems unlikely based on weights seen to date, but who knows
    float32_info = torch.finfo(torch.float32)
    final_state_dict = {}
    for k, v in avg_state_dict.items():
        v = v.clamp(float32_info.min, float32_info.max)
        final_state_dict[k] = v.to(dtype=torch.float32)

    aver_checkpoint = {"generator": final_state_dict}
    model_id = "_".join(map(str, STEPS_TO_AVER_HITRATE + STEPS_TO_AVER_MAP))
    AVER_CHECKPOINTS_PATH.mkdir(exist_ok=True, parents=True)
    torch.save(aver_checkpoint, AVER_CHECKPOINTS_PATH / f"aver_g-{model_id}")


def get_checkpoints_paths(all_checkpoints_path: Path, steps_to_aver: list[int]):
    checkpoints_paths = list[Path]()
    for step in steps_to_aver:
        model_id = str(step).zfill(8)
        checkpoints_paths.append(all_checkpoints_path / f"g_{model_id}")
    return checkpoints_paths


if __name__ == "__main__":
    main()
