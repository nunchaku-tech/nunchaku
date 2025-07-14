<div align="center" id="nunchaku_logo">
  <img src="https://raw.githubusercontent.com/mit-han-lab/nunchaku/477953fa1dd6f082fbec201cea7c7430117a810e/assets/nunchaku.svg" alt="logo" width="220"></img>
</div>
<h3 align="center">
<a href="http://arxiv.org/abs/2411.05007"><b>Paper</b></a> | <a href="https://nunchaku.tech/docs/nunchaku/"><b>Docs</b></a> | <a href="https://hanlab.mit.edu/projects/svdquant"><b>Website</b></a> | <a href="https://hanlab.mit.edu/blog/svdquant"><b>Blog</b></a> | <a href="https://svdquant.mit.edu"><b>Demo</b></a> | <a href="https://huggingface.co/collections/mit-han-lab/nunchaku-6837e7498f680552f7bbb5ad"><b>HuggingFace</b></a> | <a href="https://modelscope.cn/collections/Nunchaku-519fed7f9de94e"><b>ModelScope</b></a> | <a href="https://github.com/mit-han-lab/ComfyUI-nunchaku"><b>ComfyUI</b></a>
</h3>

<h3 align="center">
<a href="README.md"><b>English</b></a> | <a href="README_ZH.md"><b>中文</b></a>
</h3>

**Nunchaku** is a high-performance inference engine optimized for 4-bit neural networks, as introduced in our paper [SVDQuant](http://arxiv.org/abs/2411.05007). For the underlying quantization library, check out [DeepCompressor](https://github.com/mit-han-lab/deepcompressor).

Join our user groups on [**Slack**](https://join.slack.com/t/nunchaku/shared_invite/zt-3170agzoz-NgZzWaTrEj~n2KEV3Hpl5Q), [**Discord**](https://discord.gg/Wk6PnwX9Sm) and [**WeChat**](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/wechat.jpg) to engage in discussions with the community! More details can be found [here](https://github.com/mit-han-lab/nunchaku/issues/149). If you have any questions, run into issues, or are interested in contributing, don’t hesitate to reach out!

## News

- **[2025-07-13]** 🚀 The official [**Nunchaku documentation**](https://nunchaku.tech/docs/nunchaku/) is now live! Explore comprehensive guides and resources to help you get started.
- **[2025-06-29]** 🔥 Support **FLUX.1-Kontext**! Try out our [example script](./examples/flux.1-kontext-dev.py) to see it in action! Our demo is available at this [link](https://svdquant.mit.edu/kontext/)!
- **[2025-06-01]** 🚀 **Release v0.3.0!** This update adds support for multiple-batch inference, [**ControlNet-Union-Pro 2.0**](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0), initial integration of [**PuLID**](https://github.com/ToTheBeginning/PuLID), and introduces [**Double FB Cache**](examples/flux.1-dev-double_cache.py). You can now load Nunchaku FLUX models as a single file, and our upgraded [**4-bit T5 encoder**](https://huggingface.co/mit-han-lab/nunchaku-t5) now matches **FP8 T5** in quality!
- **[2025-04-16]** 🎥 Released tutorial videos in both [**English**](https://youtu.be/YHAVe-oM7U8?si=cM9zaby_aEHiFXk0) and [**Chinese**](https://www.bilibili.com/video/BV1BTocYjEk5/?share_source=copy_web&vd_source=8926212fef622f25cc95380515ac74ee) to assist installation and usage.
- **[2025-04-09]** 📢 Published the [April roadmap](https://github.com/mit-han-lab/nunchaku/issues/266) and an [FAQ](https://github.com/mit-han-lab/nunchaku/discussions/262) to help the community get started and stay up to date with Nunchaku’s development.
- **[2025-04-05]** 🚀 **Nunchaku v0.2.0 released!** This release brings [**multi-LoRA**](examples/flux.1-dev-multiple-lora.py) and [**ControlNet**](examples/flux.1-dev-controlnet-union-pro.py) support with even faster performance powered by [**FP16 attention**](#fp16-attention) and [**First-Block Cache**](#first-block-cache). We've also added compatibility for [**20-series GPUs**](examples/flux.1-dev-turing.py) — Nunchaku is now more accessible than ever!

<details>
<summary>More</summary>

- **[2025-03-07]** 🚀 **Nunchaku v0.1.4 Released!** We've supported [4-bit text encoder and per-layer CPU offloading](#Low-Memory-Inference), reducing FLUX's minimum memory requirement to just **4 GiB** while maintaining a **2–3× speedup**. This update also fixes various issues related to resolution, LoRA, pin memory, and runtime stability. Check out the release notes for full details!
- **[2025-02-20]** 🚀 **Support NVFP4 precision on NVIDIA RTX 5090!** NVFP4 delivers superior image quality compared to INT4, offering **~3× speedup** on the RTX 5090 over BF16. Learn more in our [blog](https://hanlab.mit.edu/blog/svdquant-nvfp4), checkout [`examples`](./examples) for usage and try [our demo](https://svdquant.mit.edu/flux1-schnell/) online!
- **[2025-02-18]** 🔥 [**Customized LoRA conversion**](#Customized-LoRA) and [**model quantization**](#Customized-Model-Quantization) instructions are now available! **[ComfyUI](./comfyui)** workflows now support **customized LoRA**, along with **FLUX.1-Tools**!
- **[2025-02-11]** 🎉 **[SVDQuant](http://arxiv.org/abs/2411.05007) has been selected as a ICLR 2025 Spotlight! FLUX.1-tools Gradio demos are now available!** Check [here](#gradio-demos) for the usage details! Our new [depth-to-image demo](https://svdquant.mit.edu/flux1-depth-dev/) is also online—try it out!
- **[2025-02-04]** **🚀 4-bit [FLUX.1-tools](https://blackforestlabs.ai/flux-1-tools/) is here!** Enjoy a **2-3× speedup** over the original models. Check out the [examples](./examples) for usage. **ComfyUI integration is coming soon!**
- **[2025-01-23]** 🚀 **4-bit [SANA](https://nvlabs.github.io/Sana/) support is here!** Experience a 2-3× speedup compared to the 16-bit model. Check out the [usage example](examples/sana1.6b_pag.py) and the [deployment guide](app/sana/t2i) for more details. Explore our live demo at [svdquant.mit.edu](https://svdquant.mit.edu)!
- **[2025-01-22]** 🎉 [**SVDQuant**](http://arxiv.org/abs/2411.05007) has been accepted to **ICLR 2025**!
- **[2024-12-08]** Support [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Please check [mit-han-lab/ComfyUI-nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku) for the usage.
- **[2024-11-07]** 🔥 Our latest **W4A4** Diffusion model quantization work [**SVDQuant**](https://hanlab.mit.edu/projects/svdquant) is publicly released! Check [**DeepCompressor**](https://github.com/mit-han-lab/deepcompressor) for the quantization library.

</details>

## Overview

![teaser](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/teaser.jpg)
**Nunchaku** is a high-performance inference engine for low-bit neural networks. It implements **SVDQuant**, a post-training quantization technique for 4-bit weights and activations that well maintains visual fidelity. On 12B FLUX.1-dev, it achieves 3.6× memory reduction compared to the BF16 model. By eliminating CPU offloading, it offers 8.7× speedup over the 16-bit model when on a 16GB laptop 4090 GPU, 3× faster than the NF4 W4A16 baseline. On PixArt-∑, it demonstrates significantly superior visual quality over other W4A4 or even W4A8 baselines. "E2E" means the end-to-end latency including the text encoder and VAE decoder.

**SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models**<br>
[Muyang Li](https://lmxyy.me)\*, [Yujun Lin](https://yujunlin.com)\*, [Zhekai Zhang](https://hanlab.mit.edu/team/zhekai-zhang)\*, [Tianle Cai](https://www.tianle.website/#/), [Xiuyu Li](https://xiuyuli.com), [Junxian Guo](https://github.com/JerryGJX), [Enze Xie](https://xieenze.github.io), [Chenlin Meng](https://cs.stanford.edu/~chenlin/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), and [Song Han](https://hanlab.mit.edu/songhan) <br>
*MIT, NVIDIA, CMU, Princeton, UC Berkeley, SJTU, and Pika Labs* <br>

https://github.com/user-attachments/assets/fdd4ab68-6489-4c65-8768-259bd866e8f8

## Method

#### Quantization Method -- SVDQuant

![intuition](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/intuition.gif)Overview of SVDQuant. Stage1: Originally, both the activation $\boldsymbol{X}$ and weights $\boldsymbol{W}$ contain outliers, making 4-bit quantization challenging. Stage 2: We migrate the outliers from activations to weights, resulting in the updated activation $\hat{\boldsymbol{X}}$ and weights $\hat{\boldsymbol{W}}$. While $\hat{\boldsymbol{X}}$ becomes easier to quantize, $\hat{\boldsymbol{W}}$ now becomes more difficult. Stage 3: SVDQuant further decomposes $\hat{\boldsymbol{W}}$ into a low-rank component $\boldsymbol{L}_1\boldsymbol{L}_2$ and a residual $\hat{\boldsymbol{W}}-\boldsymbol{L}_1\boldsymbol{L}_2$ with SVD. Thus, the quantization difficulty is alleviated by the low-rank branch, which runs at 16-bit precision.

#### Nunchaku Engine Design

![engine](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/engine.jpg) (a) Naïvely running low-rank branch with rank 32 will introduce 57% latency overhead due to extra read of 16-bit inputs in *Down Projection* and extra write of 16-bit outputs in *Up Projection*. Nunchaku optimizes this overhead with kernel fusion. (b) *Down Projection* and *Quantize* kernels use the same input, while *Up Projection* and *4-Bit Compute* kernels share the same output. To reduce data movement overhead, we fuse the first two and the latter two kernels together.

## Performance

![efficiency](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/efficiency.jpg)SVDQuant reduces the 12B FLUX.1 model size by 3.6× and cuts the 16-bit model's memory usage by 3.5×. With Nunchaku, our INT4 model runs 3.0× faster than the NF4 W4A16 baseline on both desktop and laptop NVIDIA RTX 4090 GPUs. Notably, on the laptop 4090, it achieves a total 10.1× speedup by eliminating CPU offloading. Our NVFP4 model is also 3.1× faster than both BF16 and NF4 on the RTX 5090 GPU.

## Getting Started

- [Installation Guide](https://nunchaku.tech/docs/nunchaku/installation/installation.html)
- [Usage Tutorial](https://nunchaku.tech/docs/nunchaku/usage/basic_usage.html)
- [ComfyUI Integration](https://github.com/mit-han-lab/ComfyUI-nunchaku)
- [Custom Quantization (DeepCompressor)](https://github.com/mit-han-lab/deepcompressor)
- [Gradio Demo Apps](https://github.com/mit-han-lab/nunchaku/tree/main/app)
- [Reproduce SVDQuant Paper Results](app/flux.1/t2i)
- [Contribution Guide](https://nunchaku.tech/docs/nunchaku/developer/contribution_guide.html)

## Roadmap

Please check [here](https://github.com/mit-han-lab/nunchaku/issues/431) for the roadmap for Summer.

## Contact Us

For enterprises interested in adopting SVDQuant or Nunchaku, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at muyangli@mit.edu.

## Related Projects

- [Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models](https://arxiv.org/abs/2211.02048), NeurIPS 2022 & T-PAMI 2023
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438), ICML 2023
- [Q-Diffusion: Quantizing Diffusion Models](https://arxiv.org/abs/2302.04304), ICCV 2023
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978), MLSys 2024
- [DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models](https://arxiv.org/abs/2402.19481), CVPR 2024
- [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/abs/2405.04532), MLSys 2025
- [SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers](https://arxiv.org/abs/2410.10629), ICLR 2025
- [Radial Attention: $O(n \\log n)$ Sparse Attention with Energy Decay for Long Video Generation](https://github.com/mit-han-lab/radial-attention), ArXiv 2025

## Citation

If you find `nunchaku` useful or relevant to your research, please cite our paper:

```bibtex
@inproceedings{
  li2024svdquant,
  title={SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models},
  author={Li*, Muyang and Lin*, Yujun and Zhang*, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze and Meng, Chenlin and Zhu, Jun-Yan and Han, Song},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

## Acknowledgments

We thank MIT-IBM Watson AI Lab, MIT and Amazon Science Hub, MIT AI Hardware Program, National Science Foundation, Packard Foundation, Dell, LG, Hyundai, and Samsung for supporting this research. We thank NVIDIA for donating the DGX server.

We use [img2img-turbo](https://github.com/GaParmar/img2img-turbo) to train the sketch-to-image LoRA. Our text-to-image and image-to-image UI is built upon [playground-v.25](https://huggingface.co/spaces/playgroundai/playground-v2.5/blob/main/app.py) and [img2img-turbo](https://github.com/GaParmar/img2img-turbo/blob/main/gradio_sketch2image.py), respectively. Our safety checker is borrowed from [hart](https://github.com/mit-han-lab/hart).

Nunchaku is also inspired by many open-source libraries, including (but not limited to) [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm), [QServe](https://github.com/mit-han-lab/qserve), [AWQ](https://github.com/mit-han-lab/llm-awq), [FlashAttention-2](https://github.com/Dao-AILab/flash-attention), and [Atom](https://github.com/efeslab/Atom).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mit-han-lab/nunchaku&type=Date)](https://www.star-history.com/#mit-han-lab/nunchaku&Date)
