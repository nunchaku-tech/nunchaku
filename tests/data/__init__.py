import os
import random

import datasets
import yaml

from nunchaku.utils import fetch_or_download

__all__ = ["get_dataset", "load_dataset_yaml"]


def load_dataset_yaml(meta_path: str, max_dataset_size: int = -1, repeat: int = 4) -> dict:
    meta = yaml.safe_load(open(meta_path, "r"))
    names = list(meta.keys())
    if max_dataset_size > 0:
        random.Random(0).shuffle(names)
        names = names[:max_dataset_size]
        names = sorted(names)

    ret = {"filename": [], "prompt": [], "meta_path": []}
    idx = 0
    for name in names:
        prompt = meta[name]
        for j in range(repeat):
            ret["filename"].append(f"{name}-{j}")
            ret["prompt"].append(prompt)
            ret["meta_path"].append(meta_path)
            idx += 1
    return ret


def get_dataset(
    name: str,
    config_name: str | None = None,
    split: str = "train",
    return_gt: bool = False,
    max_dataset_size: int = 5000,
) -> datasets.Dataset:
    prefix = os.path.dirname(__file__)
    kwargs = {
        "name": config_name,
        "split": split,
        "trust_remote_code": True,
        "token": True,
        "max_dataset_size": max_dataset_size,
    }
    path = os.path.join(prefix, f"{name}")
    if name == "MJHQ":
        dataset = datasets.load_dataset(path, return_gt=return_gt, **kwargs)
    elif name == "MJHQ-control":
        kwargs["name"] = "MJHQ-control"
        dataset = datasets.load_dataset(os.path.join(prefix, "MJHQ"), return_gt=return_gt, **kwargs)
    else:
        dataset = datasets.Dataset.from_dict(
            load_dataset_yaml(
                fetch_or_download(f"mit-han-lab/svdquant-datasets/{name}.yaml", repo_type="dataset"),
                max_dataset_size=max_dataset_size,
                repeat=1,
            ),
            features=datasets.Features(
                {
                    "filename": datasets.Value("string"),
                    "prompt": datasets.Value("string"),
                    "meta_path": datasets.Value("string"),
                }
            ),
        )
    return dataset
