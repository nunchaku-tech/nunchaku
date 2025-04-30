# Nunchaku INT4 FLUX.1 Inpainting Demo

![demo](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/app/flux.1/fill/assets/demo.jpg)

This interactive Gradio application allows you to interactively inpaint an uploaded image based on a text prompt. The base model is [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev). To launch the application, run:

```shell
python run_gradio.py
```

* The demo loads the Gemma-2B model as a safety checker by default. To disable this feature, use `--no-safety-checker`.
* To further reduce GPU memory usage, you can enable the W4A16 text encoder by specifying `--use-qencoder`.
* By default, we use our INT4 model. Use  `-p bf16` to switch to the BF16 model.
