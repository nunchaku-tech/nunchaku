# Nunchaku INT4 FLUX.1 Sketch-to-Image Demo

![demo](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/app/flux.1/sketch/assets/demo.jpg)

This interactive Gradio application transforms your drawing scribbles into realistic images given a text prompt. The base model is [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) with the [pix2pix-Turbo](https://github.com/GaParmar/img2img-turbo) sketch LoRA.

To launch the application, simply run:

```shell
python run_gradio.py
```

* The demo loads the Gemma-2B model as a safety checker by default. To disable this feature, use `--no-safety-checker`.
* To further reduce GPU memory usage, you can enable the W4A16 text encoder by specifying `--use-qencoder`.
* By default, we use our INT4 model. Use  `-p bf16` to switch to the BF16 model.
