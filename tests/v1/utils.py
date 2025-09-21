import os
from tqdm import tqdm
import math
from diffusers.utils import load_image


def run_pipeline(dataset, batch_size: int, task: str, pipeline: FluxPipeline, save_dir: str, forward_kwargs: dict = {}):
    os.makedirs(save_dir, exist_ok=True)
    pipeline.set_progress_bar_config(desc="Sampling", leave=False, dynamic_ncols=True, position=1)

    if task == "canny":
        processor = CannyDetector()
    elif task == "depth":
        processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
    elif task == "redux":
        processor = FluxPriorReduxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16
        ).to("cuda")
    else:
        assert task in ["t2i", "fill"]
        processor = None

    for row in tqdm(
        dataset.iter(batch_size=batch_size, drop_last_batch=False),
        desc="Batch",
        total=math.ceil(len(dataset) // batch_size),
        position=0,
        leave=False,
    ):
        filenames = row["filename"]
        prompts = row["prompt"]

        _forward_kwargs = {k: v for k, v in forward_kwargs.items()}

        if task == "canny":
            assert forward_kwargs.get("height", 1024) == 1024
            assert forward_kwargs.get("width", 1024) == 1024
            control_images = []
            for canny_image_path in row["canny_image_path"]:
                control_image = load_image(canny_image_path)
                control_image = processor(
                    control_image,
                    low_threshold=50,
                    high_threshold=200,
                    detect_resolution=1024,
                    image_resolution=1024,
                )
                control_images.append(control_image)
            _forward_kwargs["control_image"] = control_images
        elif task == "depth":
            control_images = []
            for depth_image_path in row["depth_image_path"]:
                control_image = load_image(depth_image_path)
                control_image = processor(control_image)[0].convert("RGB")
                control_images.append(control_image)
            _forward_kwargs["control_image"] = control_images
        elif task == "fill":
            images, mask_images = [], []
            for image_path, mask_image_path in zip(row["image_path"], row["mask_image_path"]):
                image = load_image(image_path)
                mask_image = load_image(mask_image_path)
                images.append(image)
                mask_images.append(mask_image)
            _forward_kwargs["image"] = images
            _forward_kwargs["mask_image"] = mask_images
        elif task == "redux":
            images = []
            for image_path in row["image_path"]:
                image = load_image(image_path)
                images.append(image)
            _forward_kwargs.update(processor(images))

        seeds = [hash_str_to_int(filename) for filename in filenames]
        generators = [torch.Generator().manual_seed(seed) for seed in seeds]
        if task == "redux":
            images = pipeline(generator=generators, **_forward_kwargs).images
        else:
            images = pipeline(prompts, generator=generators, **_forward_kwargs).images
        for i, image in enumerate(images):
            filename = filenames[i]
            image.save(os.path.join(save_dir, f"{filename}.png"))
        torch.cuda.empty_cache()
