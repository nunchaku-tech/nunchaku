<div align="center" id="nunchaku_logo">
  <img src="https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/nunchaku/assets/nunchaku.svg" alt="logo" width="220"></img>
</div>
<h3 align="center">
<a href="http://arxiv.org/abs/2411.05007"><b>论文</b></a> | <a href="https://nunchaku.tech/docs/nunchaku/"><b>文档</b></a> | <a href="https://hanlab.mit.edu/projects/svdquant"><b>官网</b></a> | <a href="https://hanlab.mit.edu/blog/svdquant"><b>博客</b></a> | <a href="https://svdquant.mit.edu"><b>演示</b></a> | <a href="https://huggingface.co/nunchaku-tech"><b>Hugging Face</b></a> | <a href="https://modelscope.cn/organization/nunchaku-tech"><b>魔搭社区</b></a> | <a href="https://github.com/nunchaku-tech/ComfyUI-nunchaku"><b>ComfyUI</b></a>
</h3>

<h3 align="center">
<a href="README.md"><b>English</b></a> | <a href="README_ZH.md"><b>中文</b></a>
</h3>

**Nunchaku** 是一款为 4-bit 神经网络优化的高性能推理引擎，详见我们的论文 [SVDQuant](http://arxiv.org/abs/2411.05007)。底层量化库请参考 [DeepCompressor](https://github.com/nunchaku-tech/deepcompressor)。

欢迎加入我们的用户群：[**Slack**](https://join.slack.com/t/nunchaku/shared_invite/zt-3170agzoz-NgZzWaTrEj~n2KEV3Hpl5Q)、[**Discord**](https://discord.gg/Wk6PnwX9Sm) 和 [**微信**](https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/nunchaku/assets/wechat.jpg)，与社区交流！更多详情见 [这里](https://github.com/nunchaku-tech/nunchaku/issues/149)。如有问题、遇到 bug 或有意贡献代码，欢迎随时联系我们！

## 最新动态

- **[2025-08-27]** 🚀 发布 **4-bit [4/8步 lightning Qwen-Image](https://huggingface.co/lightx2v/Qwen-Image-Lightning)**！可在 [Hugging Face](https://huggingface.co/nunchaku-tech/nunchaku-qwen-image) 和 [ModelScope](https://modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image) 下载。使用我们的 [示例脚本](examples/v1/qwen-image-lightning.py) 开始体验。
- **[2025-07-31]** 🚀 **[FLUX.1-Krea-dev](https://www.krea.ai/blog/flux-krea-open-source-release) 已支持！** 欢迎参考我们的[示例脚本](./examples/flux.1-krea-dev.py)快速上手。
- **[2025-07-13]** 🚀 官方 [**Nunchaku 文档**](https://nunchaku.tech/docs/nunchaku/) 上线！欢迎查阅详细的入门指南和资源。
- **[2025-06-29]** 🔥 支持 **FLUX.1-Kontext**！可参考我们的[示例脚本](./examples/flux.1-kontext-dev.py)体验，在线演示见[此处](https://svdquant.mit.edu/kontext/)！
- **[2025-06-01]** 🚀 **v0.3.0 发布！** 本次更新支持多 batch 推理、[**ControlNet-Union-Pro 2.0**](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0)、初步集成 [**PuLID**](https://github.com/ToTheBeginning/PuLID)，并引入 [**双 FB Cache**](examples/flux.1-dev-double_cache.py)。现已支持单文件加载 FLUX 模型，升级后的 [**4-bit T5 编码器**](https://huggingface.co/nunchaku-tech/nunchaku-t5) 质量媲美 **FP8 T5**！

<details>
<summary>更多历史</summary>

- **[2025-04-16]** 🎥 发布中英文[**安装与使用教程视频**](https://youtu.be/YHAVe-oM7U8?si=cM9zaby_aEHiFXk0)（[**B站**](https://www.bilibili.com/video/BV1BTocYjEk5/?share_source=copy_web&vd_source=8926212fef622f25cc95380515ac74ee)）。
- **[2025-04-09]** 📢 发布 [四月路线图](https://github.com/nunchaku-tech/nunchaku/issues/266) 及 [FAQ](https://github.com/nunchaku-tech/nunchaku/discussions/262)，助力社区快速上手并了解最新进展。
- **[2025-04-05]** 🚀 **Nunchaku v0.2.0 发布！** 本次更新带来 [**多 LoRA**](examples/flux.1-dev-multiple-lora.py) 和 [**ControlNet**](examples/flux.1-dev-controlnet-union-pro.py) 支持，并通过 [**FP16 attention**](#fp16-attention) 和 [**First-Block Cache**](#first-block-cache) 实现更快推理。现已兼容 [**20 系显卡**](examples/flux.1-dev-turing.py) —— Nunchaku 更易用！
- **[2025-03-07]** 🚀 **Nunchaku v0.1.4 发布！** 支持 [4-bit 文本编码器和逐层 CPU 下放](#Low-Memory-Inference)，将 FLUX 最低显存需求降至 **4 GiB**，同时实现 **2–3× 加速**。本次还修复了分辨率、LoRA、pin memory 和稳定性等问题，详见发布说明！
- **[2025-02-20]** 🚀 **RTX 5090 支持 NVFP4 精度！** NVFP4 相比 INT4 画质更佳，在 RTX 5090 上比 BF16 快 **~3×**。详情见[博客](https://hanlab.mit.edu/blog/svdquant-nvfp4)，用法见 [`examples`](./examples)，在线体验[点此](https://svdquant.mit.edu/flux1-schnell/)！
- **[2025-02-18]** 🔥 [**自定义 LoRA 转换**](#Customized-LoRA) 和 [**模型量化**](#Customized-Model-Quantization) 教程上线！**[ComfyUI](./comfyui)** 工作流现已支持 **自定义 LoRA** 及 **FLUX.1-Tools**！
- **[2025-02-11]** 🎉 **[SVDQuant](http://arxiv.org/abs/2411.05007) 入选 ICLR 2025 Spotlight！FLUX.1-tools Gradio 演示上线！** 详情见 [这里](#gradio-demos)。全新 [depth-to-image 演示](https://svdquant.mit.edu/flux1-depth-dev/) 也已上线，欢迎体验！
- **[2025-02-04]** **🚀 4-bit [FLUX.1-tools](https://blackforestlabs.ai/flux-1-tools/) 发布！** 推理速度比原模型快 **2-3×**。用法见 [examples](./examples)。**ComfyUI 集成即将上线！**
- **[2025-01-23]** 🚀 **4-bit [SANA](https://nvlabs.github.io/Sana/) 支持！** 推理速度比 16-bit 模型快 2-3×。用法见 [示例](examples/sana1.6b_pag.py) 和 [部署指南](app/sana/t2i)。在线体验 [svdquant.mit.edu](https://svdquant.mit.edu)！
- **[2025-01-22]** 🎉 [**SVDQuant**](http://arxiv.org/abs/2411.05007) 被 **ICLR 2025** 录用！
- **[2024-12-08]** 支持 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)。用法见 [ComfyUI-nunchaku](https://github.com/nunchaku-tech/ComfyUI-nunchaku)。
- **[2024-11-07]** 🔥 最新 **W4A4** Diffusion 量化工作 [**SVDQuant**](https://hanlab.mit.edu/projects/svdquant) 正式发布！量化库见 [**DeepCompressor**](https://github.com/nunchaku-tech/deepcompressor)。

</details>

## 总览

![teaser](https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/nunchaku/assets/teaser.jpg)
**Nunchaku** 是一款面向低比特神经网络的高性能推理引擎。其实现了 **SVDQuant**，一种针对 4-bit 权重和激活的后训练量化技术，能很好地保持视觉质量。在 12B FLUX.1-dev 上，相比 BF16 模型实现了 3.6× 显存缩减。通过消除 CPU 下放，在 16GB 笔记本 4090 GPU 上比 16-bit 模型快 8.7×，比 NF4 W4A16 基线快 3×。在 PixArt-∑ 上，视觉质量显著优于其他 W4A4 甚至 W4A8 基线。"E2E" 指包括文本编码器和 VAE 解码器的端到端延迟。

**SVDQuant: 通过低秩分支吸收异常值，实现 4-bit Diffusion 模型**<br>
[Muyang Li](https://lmxyy.me)\*，[Yujun Lin](https://yujunlin.com)\*，[Zhekai Zhang](https://hanlab.mit.edu/team/zhekai-zhang)\*，[Tianle Cai](https://www.tianle.website/#/)，[Xiuyu Li](https://xiuyuli.com)，[Junxian Guo](https://github.com/JerryGJX)，[Enze Xie](https://xieenze.github.io)，[Chenlin Meng](https://cs.stanford.edu/~chenlin/)，[Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)，[Song Han](https://hanlab.mit.edu/songhan) <br>
*MIT, NVIDIA, CMU, Princeton, UC Berkeley, SJTU, Pika Labs* <br>

https://github.com/user-attachments/assets/fdd4ab68-6489-4c65-8768-259bd866e8f8

## 方法

#### 量化方法 -- SVDQuant

![intuition](https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/nunchaku/assets/intuition.gif)SVDQuant 概览。阶段1：原始激活 $\boldsymbol{X}$ 和权重 $\boldsymbol{W}$ 都包含异常值，导致 4-bit 量化困难。阶段2：我们将异常值从激活迁移到权重，得到更新后的激活 $\hat{\boldsymbol{X}}$ 和权重 $\hat{\boldsymbol{W}}$。此时 $\hat{\boldsymbol{X}}$ 更易量化，但 $\hat{\boldsymbol{W}}$ 更难。阶段3：SVDQuant 进一步将 $\hat{\boldsymbol{W}}$ 分解为低秩分支 $\boldsymbol{L}_1\boldsymbol{L}_2$ 和残差 $\hat{\boldsymbol{W}}-\boldsymbol{L}_1\boldsymbol{L}_2$。低秩分支用 16-bit 精度运行，从而缓解量化难度。

#### Nunchaku 引擎设计

![engine](https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/nunchaku/assets/engine.jpg) (a) 直接用 rank 32 跑低秩分支会带来 57% 延迟开销，因需额外读写 16-bit 输入/输出。Nunchaku 通过内核融合优化此开销。(b) *Down Projection* 和 *Quantize* 内核输入相同，*Up Projection* 和 *4-Bit Compute* 内核输出相同。为减少数据搬运，Nunchaku 将前两者和后两者分别融合。

## 性能

![efficiency](https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/nunchaku/assets/efficiency.jpg)SVDQuant 将 12B FLUX.1 模型体积缩小 3.6×，显存占用降至 16-bit 模型的 1/3.5。Nunchaku 的 INT4 模型在桌面和笔记本 4090 上比 NF4 W4A16 基线快 3.0×。在笔记本 4090 上，通过消除 CPU 下放，总加速比达 10.1×。NVFP4 模型在 RTX 5090 上也比 BF16 和 NF4 快 3.1×。

## 快速上手

- [安装指南](https://nunchaku.tech/docs/nunchaku/installation/installation.html)
- [使用教程](https://nunchaku.tech/docs/nunchaku/usage/basic_usage.html)
- [ComfyUI 插件：ComfyUI-nunchaku](https://github.com/nunchaku-tech/ComfyUI-nunchaku)
- [自定义模型量化：DeepCompressor](https://github.com/nunchaku-tech/deepcompressor)
- [Gradio 演示应用](https://github.com/nunchaku-tech/nunchaku/tree/main/app)
- [复现 SVDQuant 论文结果](app/flux.1/t2i)
- [API 参考](https://nunchaku.tech/docs/nunchaku/python_api/nunchaku.html)
- [贡献指南](https://nunchaku.tech/docs/nunchaku/developer/contribution_guide.html)
- [常见问题 FAQ](https://nunchaku.tech/docs/nunchaku/faq/faq.html)

## 路线图

暑期开发计划见 [这里](https://github.com/nunchaku-tech/nunchaku/issues/431)。

## 联系我们

如有企业合作、技术咨询、赞助或合作意向，请联系 muyangli@mit.edu。

## 相关项目

- [Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models](https://arxiv.org/abs/2211.02048), NeurIPS 2022 & T-PAMI 2023
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438), ICML 2023
- [Q-Diffusion: Quantizing Diffusion Models](https://arxiv.org/abs/2302.04304), ICCV 2023
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978), MLSys 2024
- [DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models](https://arxiv.org/abs/2402.19481), CVPR 2024
- [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/abs/2405.04532), MLSys 2025
- [SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers](https://arxiv.org/abs/2410.10629), ICLR 2025
- [Radial Attention: $O(n \log n)$ Sparse Attention with Energy Decay for Long Video Generation](https://github.com/mit-han-lab/radial-attention), ArXiv 2025

## 引用

如果您觉得 `nunchaku` 对您的研究有用或相关，请引用我们的论文：

```bibtex
@inproceedings{
  li2024svdquant,
  title={SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models},
  author={Li*, Muyang and Lin*, Yujun and Zhang*, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze and Meng, Chenlin and Zhu, Jun-Yan and Han, Song},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

## 致谢

我们感谢 MIT-IBM Watson AI Lab、MIT 和 Amazon Science Hub、MIT AI Hardware Program、National Science Foundation、Packard Foundation、Dell、LG、现代和三星对本研究的支持。我们感谢 NVIDIA 捐赠的 DGX 服务器。我们感谢 [First Intelligence](https://www.first-intelligence.com/) 和 [Yotta Labs](https://www.yottalabs.ai/) 赞助我们的计算资源。

我们使用 [img2img-turbo](https://github.com/GaParmar/img2img-turbo) 训练草图到图像的 LoRA。我们的文本到图像和图像到图像 UI 分别基于 [playground-v.25](https://huggingface.co/spaces/playgroundai/playground-v2.5/blob/main/app.py) 和 [img2img-turbo](https://github.com/GaParmar/img2img-turbo/blob/main/gradio_sketch2image.py) 构建。我们的安全检查器来自 [hart](https://github.com/mit-han-lab/hart)。

Nunchaku 还受到许多开源库的启发，包括（但不限于）[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)、[vLLM](https://github.com/vllm-project/vllm)、[QServe](https://github.com/mit-han-lab/qserve)、[AWQ](https://github.com/mit-han-lab/llm-awq)、[FlashAttention-2](https://github.com/Dao-AILab/flash-attention) 和 [Atom](https://github.com/efeslab/Atom)。

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=nunchaku-tech/nunchaku&type=Date)](https://www.star-history.com/#nunchaku-tech/nunchaku&Date)
