import argparse
import json
import os

from data import get_dataset
from metrics.fid import compute_fid
from metrics.image_reward import compute_image_reward
from metrics.multimodal import compute_image_multimodal_metrics
from metrics.similarity import compute_image_similarity_metrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_roots", type=str, nargs="*")
    parser.add_argument("-o", "--output-path", type=str, default="metrics.json", help="Image output path")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    assert len(args.input_roots) > 0
    assert len(args.input_roots) <= 2

    image_root1 = args.input_roots[0]
    if len(args.input_roots) > 1:
        image_root2 = args.input_roots[1]
    else:
        image_root2 = None

    results = {}
    dataset_names = sorted(os.listdir(image_root1))
    for dataset_name in dataset_names:
        if image_root2 is not None and dataset_name not in os.listdir(image_root2):
            continue
        print("Results for dataset:", dataset_name)
        results[dataset_name] = {}
        dataset = get_dataset(name=dataset_name, return_gt=True, max_dataset_size=1024)
        fid = compute_fid(ref_dirpath_or_dataset=dataset, gen_dirpath=os.path.join(image_root1, dataset_name))
        results[dataset_name]["fid"] = fid
        print("FID:", fid)
        multimodal_metrics = compute_image_multimodal_metrics(
            ref_dataset=dataset, gen_dirpath=os.path.join(image_root1, dataset_name)
        )
        results[dataset_name].update(multimodal_metrics)
        for k, v in multimodal_metrics.items():
            print(f"{k}:", v)
        image_reward = compute_image_reward(ref_dataset=dataset, gen_dirpath=os.path.join(image_root1, dataset_name))
        results[dataset_name].update(image_reward)
        for k, v in image_reward.items():
            print(f"{k}:", v)

        if image_root2 is not None and os.path.exists(os.path.join(image_root2, dataset_name)):
            similarity_results = compute_image_similarity_metrics(
                os.path.join(image_root1, dataset_name), os.path.join(image_root2, dataset_name)
            )
            results[dataset_name].update(similarity_results)
            for k, v in similarity_results.items():
                print(f"{k}:", v)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
