import gc
import os
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from tqdm import trange

from ..utils import hash_str_to_int


def run_pipeline(
    dataset: list[dict],
    batch_size: int,
    pipeline: DiffusionPipeline,
    save_dir: os.PathLike[str],
    forward_kwargs: dict = {},
):
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    assert isinstance(save_dir, Path)
    save_dir.mkdir(parents=True, exist_ok=True)

    pipeline.set_progress_bar_config(desc="Sampling", leave=False, dynamic_ncols=True, position=1)
    for batch_idx in trange(len(dataset) // batch_size, desc="Batch", position=0, leave=False):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch = dataset[start_idx:end_idx]

        filenames = [_["filename"] for _ in batch]
        generators = [torch.Generator().manual_seed(hash_str_to_int(filename)) for filename in filenames]
        _forward_kwargs = {k: v for k, v in forward_kwargs.items()}
        _forward_kwargs["generator"] = generators
        for k in batch[0].keys():
            if k == "filename":
                continue
            _forward_kwargs[k] = [_[k] for _ in batch]
        images = pipeline(**_forward_kwargs).images
        for i, image in enumerate(images):
            filename = filenames[i]
            image.save(os.path.join(save_dir, f"{filename}.png"))
    gc.collect()
    torch.cuda.empty_cache()
