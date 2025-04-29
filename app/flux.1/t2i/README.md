# Nunchaku INT4 FLUX.1 Models

## Text-to-Image Gradio Demo

![demo](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/app/flux.1/t2i/assets/demo.jpg)

This interactive Gradio application can generate an image based on your provided text prompt. The base model can be either [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) or [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev).

To launch the application, simply run:

```shell
python run_gradio.py
```

* The demo also defaults to the FLUX.1-schnell model. To switch to the FLUX.1-dev model, use `-m dev`.
* By default, the Gemma-2B model is loaded as a safety checker. To disable this feature and save GPU memory, use `--no-safety-checker`.
* To further reduce GPU memory usage, you can enable the W4A16 text encoder by specifying `--use-qencoder`.
* By default, only the INT4 DiT is loaded. Use `-p int4 bf16` to add a BF16 DiT for side-by-side comparison, or `-p bf16` to load only the BF16 model.

## Command Line Inference

We provide a script, [generate.py](generate.py), that generates an image from a text prompt directly from the command line, similar to the demo. Simply run:

```shell
python generate.py --prompt "You Text Prompt"
```

* The generated image will be saved as `output.png` by default. You can specify a different path using the `-o` or `--output-path` options.
* The script defaults to using the FLUX.1-schnell model. To switch to the FLUX.1-dev model, use `-m dev`.
* By default, the script uses our INT4 model. To use the BF16 model instead, specify `-p bf16`.
* You can specify `--use-qencoder` to use our W4A16 text encoder.
* You can adjust the number of inference steps and guidance scale with `-t` and `-g`, respectively. For the FLUX.1-schnell model, the defaults are 4 steps and a guidance scale of 0; for the FLUX.1-dev model, the defaults are 50 steps and a guidance scale of 3.5.

* When using the FLUX.1-dev model, you also have the option to load a LoRA adapter with `--lora-name`. Available choices are `None`, [`Anime`](https://huggingface.co/alvdansen/sonny-anime-fixed), [`GHIBSKY Illustration`](https://huggingface.co/aleksa-codes/flux-ghibsky-illustration), [`Realism`](https://huggingface.co/XLabs-AI/flux-RealismLora), [`Children Sketch`](https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch), and [`Yarn Art`](https://huggingface.co/linoyts/yarn_art_Flux_LoRA), with the default set to `None`. You can also specify the LoRA weight with `--lora-weight`, which defaults to 1.

## Latency Benchmark

To measure the latency of our INT4 models, use the following command:

```shell
python latency.py
```

* The script defaults to the INT4 FLUX.1-schnell model. To switch to FLUX.1-dev, use the `-m dev` option. For BF16 precision, add `-p bf16`.
* Adjust the number of inference steps and the guidance scale using `-t` and `-g`, respectively.
  - For FLUX.1-schnell, the defaults are 4 steps and a guidance scale of 0.
  - For FLUX.1-dev, the defaults are 50 steps and a guidance scale of 3.5.
* By default, the script measures the end-to-end latency for generating a single image. To measure the latency of a single DiT forward step instead, use the `--mode step` flag.
* Specify the number of warmup and test runs using `--warmup-times` and `--test-times`. The defaults are 2 warmup runs and 10 test runs.

## Quality Results

Below are the steps to reproduce the quality metrics reported in our paper. Firstly, you will need to install slightly more packages for the image quality metrics:

```shell
pip install clean-fid torchmetrics image-reward clip datasets
```

Then generate images using both the original BF16 model and our INT4 model on the [MJHQ](https://huggingface.co/datasets/playgroundai/MJHQ-30K) and [DCI](https://github.com/facebookresearch/DCI) datasets:

```shell
python evaluate.py -p int4
python evaluate.py -p bf16
```

* The commands above will generate images from FLUX.1-schnell on both datasets. Use `-m dev` to switch to FLUX.1-dev, or specify a single dataset with `-d MJHQ` or `-d DCI`.
* By default, generated images are saved to `results/$MODEL/$PRECISION`. Customize the output path using the `-o` option if desired.
* You can also adjust the number of inference steps and the guidance scale using `-t` and `-g`, respectively.
  - For FLUX.1-schnell, the defaults are 4 steps and a guidance scale of 0.
  - For FLUX.1-dev, the defaults are 50 steps and a guidance scale of 3.5.
* To accelerate the generation process, you can distribute the workload across multiple GPUs. For instance, if you have $N$ GPUs, on GPU $i (0 \le i < N)$ , you can add the options `--chunk-start $i --chunk-step $N`. This setup ensures each GPU handles a distinct portion of the workload, enhancing overall efficiency.

Finally you can compute the metrics for the images with

```shell
python get_metrics.py results/schnell/int4 results/schnell/bf16
```

Remember to replace the example paths with the actual paths to your image folders.

**Notes:**

- The script will calculate quality metrics (CLIP IQA, CLIP Score, Image Reward, FID) only for the first folder specified. Ensure the INT4 results folder is listed first.
- **Similarity Metrics**: If a second folder path is not provided, similarity metrics (LPIPS, PSNR, SSIM) will be skipped.
- **Output File**: Metric results are saved in `metrics.json` by default. Use `-o` to specify a custom output file if needed.
