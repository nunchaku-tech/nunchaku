# Nunchaku [SANA](https://nvlabs.github.io/Sana/) Models

## Text-to-Image Gradio Demo

![demo](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/app/sana/t2i/assets/demo.jpg)

This interactive Gradio application can generate an image based on your provided text prompt. The base model is [SANA-1.6B](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers).

```shell
python run_gradio.py
```

* By default, the Gemma-2B model is loaded as a safety checker. To disable this feature and save GPU memory, use `--no-safety-checker`.
* By default, only the INT4 DiT is loaded. Use `-p int4 bf16` to add a BF16 DiT for side-by-side comparison, or `-p bf16` to load only the BF16 model.

## Command Line Inference

We provide a script, [generate.py](generate.py), that generates an image from a text prompt directly from the command line, similar to the demo. Simply run:

```shell
python generate.py --prompt "You Text Prompt"
```

* The generated image will be saved as `output.png` by default. You can specify a different path using the `-o` or `--output-path` options.
* By default, the script uses our INT4 model. To use the BF16 model instead, specify `-p bf16`.
* You can adjust the number of inference steps and classifier-free guidance scale with `-t` and `-g`, respectively. The defaults are 20 steps and a guidance scale of 5.
* In addition to the classifier-free guidance, you can also adjust the [PAG guidance](https://arxiv.org/abs/2403.17377) scale with `--pag-scale`. The default is 2.

## Latency Benchmark

To measure the latency of our INT4 models, use the following command:

```shell
python latency.py
```

* Adjust the number of inference steps and the guidance scale using `-t` and `-g`, respectively. The defaults are 20 steps and a guidance scale of 5.
* You can also adjust the [PAG guidance](https://arxiv.org/abs/2403.17377) scale with `--pag-scale`. The default is 2.
* By default, the script measures the end-to-end latency for generating a single image. To measure the latency of a single DiT forward step instead, use the `--mode step` flag.
* Specify the number of warmup and test runs using `--warmup-times` and `--test-times`. The defaults are 2 warmup runs and 10 test runs.
