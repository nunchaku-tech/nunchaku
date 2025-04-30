import os

import datasets
import ImageReward as RM
import torch
from tqdm import tqdm


def compute_image_reward(
    ref_dataset: datasets.Dataset,
    gen_dirpath: str,
) -> dict[str, float]:
    scores = []
    model = RM.load("ImageReward-v1.0")
    for batch in tqdm(
        ref_dataset.iter(batch_size=1, drop_last_batch=False),
        desc=f"{ref_dataset.config_name} image reward",
        total=len(ref_dataset),
        dynamic_ncols=True,
    ):
        filename = batch["filename"][0]
        path = os.path.join(gen_dirpath, f"{filename}.png")
        prompt = batch["prompt"][0]
        with torch.inference_mode():
            score = model.score(prompt, path)
        scores.append(score)
    result = {"image_reward": sum(scores) / len(scores)}
    return result
