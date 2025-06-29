# Nunchaku INT4 FLUX.1 Kontext Demo

![demo](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/kontext.png)

This interactive Gradio application allows you to edit an image with natural language. Simply run:

```shell
python run_gradio.py
```

- To further reduce GPU memory usage, you can enable the W4A16 text encoder by specifying `--use-qencoder`.
- By default, we use our INT4 model. Use `-p bf16` to switch to the BF16 model.
