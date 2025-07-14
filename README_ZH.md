<div align="center" id="nunchaku_logo">
  <img src="https://raw.githubusercontent.com/mit-han-lab/nunchaku/477953fa1dd6f082fbec201cea7c7430117a810e/assets/nunchaku.svg" alt="logo" width="220"></img>
</div>
<h3 align="center">
<a href="http://arxiv.org/abs/2411.05007"><b>论文</b></a> | <a href="https://nunchaku.tech/docs/nunchaku/"><b>文档</b></a> | <a href="https://hanlab.mit.edu/projects/svdquant"><b>官网</b></a> | <a href="https://hanlab.mit.edu/blog/svdquant"><b>博客</b></a> | <a href="https://svdquant.mit.edu"><b>演示</b></a> | <a href="https://huggingface.co/collections/mit-han-lab/nunchaku-6837e7498f680552f7bbb5ad"><b>HuggingFace</b></a> | <a href="https://modelscope.cn/collections/Nunchaku-519fed7f9de94e"><b>ModelScope</b></a> | <a href="https://github.com/mit-han-lab/ComfyUI-nunchaku"><b>ComfyUI</b></a>
</h3>

<h3 align="center">
<a href="README.md"><b>English</b></a> | <a href="README_ZH.md"><b>中文</b></a>
</h3>

**Nunchaku**是一款专为4-bit神经网络优化的高性能推理引擎，基于我们的论文 [SVDQuant](http://arxiv.org/abs/2411.05007) 提出。底层量化库请参考 [DeepCompressor](https://github.com/mit-han-lab/deepcompressor)。

欢迎加入我们的用户群：[**Slack**](https://join.slack.com/t/nunchaku/shared_invite/zt-3170agzoz-NgZzWaTrEj~n2KEV3Hpl5Q)、[**Discord**](https://discord.gg/Wk6PnwX9Sm) 和 [**微信**](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/wechat.jpg)，与社区交流！更多详情请见[此处](https://github.com/mit-han-lab/nunchaku/issues/149)。如有任何问题、建议或贡献意向，欢迎随时联系！

## 最新动态

- **[2025-07-13]** 🚀 官方 [**Nunchaku 文档**](https://nunchaku.tech/docs/nunchaku/) 正式上线！探索全面的指南和资源，助您快速上手。
- **[2025-06-29]** 🔥 支持 **FLUX.1-Kontext**！试用我们的 [示例脚本](./examples/flux.1-kontext-dev.py) 体验效果！演示站点请访问 [链接](https://svdquant.mit.edu/kontext/)！
- **[2025-06-01]** 🚀 **发布 v0.3.0！** 此次更新新增多批次推理支持、[**ControlNet-Union-Pro 2.0**](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0)、[**PuLID**](https://github.com/ToTheBeginning/PuLID) 初步集成，以及 [**Double FB Cache**](examples/flux.1-dev-double_cache.py) 功能。现在您可以将 Nunchaku FLUX 模型加载为单个文件，升级的 [**4-bit T5 编码器**](https://huggingface.co/mit-han-lab/nunchaku-t5) 在质量上已与 **FP8 T5** 持平！
- **[2025-04-16]** 🎥 发布了[**英文**](https://youtu.be/YHAVe-oM7U8?si=cM9zaby_aEHiFXk0)和[**中文**](https://www.bilibili.com/video/BV1BTocYjEk5/?share_source=copy_web&vd_source=8926212fef622f25cc95380515ac74ee)教程视频，协助安装和使用。
- **[2025-04-09]** 📢 发布 [四月开发路线图](https://github.com/mit-han-lab/nunchaku/issues/266) 和 [常见问题解答](https://github.com/mit-han-lab/nunchaku/discussions/262)，帮助社区快速上手并了解 Nunchaku 最新进展。
- **[2025-04-05]** 🚀 **Nunchaku v0.2.0 发布！** 支持 [**多LoRA**](examples/flux.1-dev-multiple-lora.py) 和 [**ControlNet**](examples/flux.1-dev-controlnet-union-pro.py)，通过 [**FP16 attention**](#fp16-attention) 和 [**First-Block Cache**](#first-block-cache) 实现更快的推理速度。新增 [**20系显卡支持**](examples/flux.1-dev-turing.py) — Nunchaku 现在更加易于使用！

<details>
<summary>更多动态</summary>

- **[2025-03-07]** 🚀 **Nunchaku v0.1.4 发布！** 支持 [4-bit文本编码器和分层CPU offloading](#%E4%BD%8E%E6%98%BE%E5%AD%98%E6%8E%A8%E7%90%86)，FLUX最低显存需求降至 **4 GiB**，同时保持 **2–3倍加速**。此次更新还修复了分辨率、LoRA、内存锁定和运行时稳定性等问题。详情请查看发布说明！
- **[2025-02-20]** 🚀 **NVIDIA RTX 5090 支持 NVFP4 精度！** 相比 INT4，NVFP4 画质更优，在 RTX 5090 上比 BF16 快 **约3倍**。[博客详解](https://hanlab.mit.edu/blog/svdquant-nvfp4)，[示例代码](./examples) 及 [在线演示](https://svdquant.mit.edu/flux1-schnell/) 已上线！
- **[2025-02-18]** 🔥 [**自定义LoRA转换**](#%E8%87%AA%E5%AE%9A%E4%B9%89lora) 和 [**模型量化**](#%E8%87%AA%E5%AE%9A%E4%B9%89%E6%A8%A1%E5%9E%8B%E9%87%8F%E5%8C%96) 指南现已发布！**[ComfyUI](./comfyui)** 工作流现在支持 **自定义LoRA** 和 **FLUX.1-Tools**！
- **[2025-02-11]** 🎉 **[SVDQuant](http://arxiv.org/abs/2411.05007) 入选 ICLR 2025 Spotlight！FLUX.1-tools Gradio 演示已上线！** 使用详情请查看 [这里](#gradio-%E6%BC%94%E7%A4%BA)！我们新的 [深度图生成演示](https://svdquant.mit.edu/flux1-depth-dev/) 也已上线—快来试试吧！
- **[2025-02-04]** **🚀 4-bit [FLUX.1-tools](https://blackforestlabs.ai/flux-1-tools/) 来了！** 相比原模型提速 **2-3倍**。使用方法请查看 [示例](./examples)。**ComfyUI 集成即将推出！**
- **[2025-01-23]** 🚀 **4-bit [SANA](https://nvlabs.github.io/Sana/) 支持来了！** 相比16位模型提速2-3倍。查看 [使用示例](examples/sana1.6b_pag.py) 和 [部署指南](app/sana/t2i) 了解详情。体验我们的在线演示 [svdquant.mit.edu](https://svdquant.mit.edu)！
- **[2025-01-22]** 🎉 [**SVDQuant**](http://arxiv.org/abs/2411.05007) 被 **ICLR 2025** 接收！
- **[2024-12-08]** 支持 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)。使用方法请查看 [mit-han-lab/ComfyUI-nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku)。
- **[2024-11-07]** 🔥 我们最新的 **W4A4** 扩散模型量化工作 [**SVDQuant**](https://hanlab.mit.edu/projects/svdquant) 公开发布！量化库 [**DeepCompressor**](https://github.com/mit-han-lab/deepcompressor) 同步发布。

</details>

## 项目概览

![teaser](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/teaser.jpg)
**Nunchaku** 是一款专为低精度神经网络设计的高性能推理引擎。它实现了 **SVDQuant**，一种支持4-bit权重和激活的后训练量化技术，能有效保持视觉质量。在12B FLUX.1-dev模型上，相比BF16模型实现了3.6倍内存压缩。通过消除CPU offloading，在16GB笔记本RTX 4090上比16位模型快8.7倍，比NF4 W4A16基线快3倍。在PixArt-∑模型上，其视觉质量显著优于其他W4A4甚至W4A8方案。"E2E"表示包含文本编码器和VAE解码器的端到端延迟。

**SVDQuant: 通过低秩分量吸收异常值实现4-bit扩散模型量化**<br>
[Muyang Li](https://lmxyy.me)\*, [Yujun Lin](https://yujunlin.com)\*, [Zhekai Zhang](https://hanlab.mit.edu/team/zhekai-zhang)\*, [Tianle Cai](https://www.tianle.website/#/), [Xiuyu Li](https://xiuyuli.com), [Junxian Guo](https://github.com/JerryGJX), [Enze Xie](https://xieenze.github.io), [Chenlin Meng](https://cs.stanford.edu/~chenlin/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Song Han](https://hanlab.mit.edu/songhan) <br>
*麻省理工学院、英伟达、卡内基梅隆大学、普林斯顿大学、加州大学伯克利分校、上海交通大学、pika实验室* <br>

https://github.com/user-attachments/assets/fdd4ab68-6489-4c65-8768-259bd866e8f8

## 方法原理

#### 量化方法 -- SVDQuant

![intuition](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/intuition.gif)SVDQuant三阶段示意图。阶段1：原始激活 $\boldsymbol{X}$ 和权重 $\boldsymbol{W}$ 均含异常值，4-bit量化困难。阶段2：将激活异常值迁移至权重，得到更新的激活 $\hat{\boldsymbol{X}}$ 和权重 $\hat{\boldsymbol{W}}$。虽然 $\hat{\boldsymbol{X}}$ 更易量化，但 $\hat{\boldsymbol{W}}$ 变得更难量化。阶段3：SVDQuant 进一步通过 SVD 将 $\hat{\boldsymbol{W}}$ 分解为低秩分量 $\boldsymbol{L}_1\boldsymbol{L}_2$ 和残差 $\hat{\boldsymbol{W}}-\boldsymbol{L}_1\boldsymbol{L}_2$。通过16位精度运行低秩分支来缓解量化难度。

#### Nunchaku 引擎设计

![engine](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/engine.jpg) (a) 原始低秩分支（秩32）因在 *下投影* 中额外读取16位输入和在 *上投影* 中额外写入16位输出而引入57%的延迟开销。Nunchaku 通过核融合优化此开销。(b) *下投影* 和 *量化* 核使用相同输入，*上投影* 和 *4-bit计算* 核共享相同输出。为减少数据搬运开销，我们将前两个核和后两个核分别融合。

## 性能表现

![efficiency](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/efficiency.jpg)SVDQuant 将12B FLUX.1模型体积减少了3.6倍，并将16位模型的显存使用量减少了3.5倍。借助 Nunchaku，我们的 INT4 模型在桌面和笔记本 NVIDIA RTX 4090 GPU 上比 NF4 W4A16 基线快3.0倍。值得注意的是，在笔记本4090上，通过消除CPU offloading，总体加速达到了10.1倍。我们的 NVFP4 模型在 RTX 5090 GPU 上也比 BF16 和 NF4 快3.1倍。

## 快速开始

- [安装指南](https://nunchaku.tech/docs/nunchaku/installation/installation.html)
- [使用教程](https://nunchaku.tech/docs/nunchaku/usage/basic_usage.html)
- [ComfyUI 集成](https://github.com/mit-han-lab/ComfyUI-nunchaku)
- [自定义量化 (DeepCompressor)](https://github.com/mit-han-lab/deepcompressor)
- [Gradio 演示应用](https://github.com/mit-han-lab/nunchaku/tree/main/app)
- [复现 SVDQuant 论文结果](app/flux.1/t2i)
- [贡献指南](https://nunchaku.tech/docs/nunchaku/developer/contribution_guide.html)

## 路线图

请查看 [这里](https://github.com/mit-han-lab/nunchaku/issues/431) 获取夏季开发路线图。

## 联系我们

有意采用 SVDQuant 或 Nunchaku 的企业，包括技术咨询、赞助机会或合作咨询，请联系我们：muyangli@mit.edu。

## 相关项目

- [Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models](https://arxiv.org/abs/2211.02048), NeurIPS 2022 & T-PAMI 2023
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438), ICML 2023
- [Q-Diffusion: Quantizing Diffusion Models](https://arxiv.org/abs/2302.04304), ICCV 2023
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978), MLSys 2024
- [DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models](https://arxiv.org/abs/2402.19481), CVPR 2024
- [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/abs/2405.04532), MLSys 2025
- [SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers](https://arxiv.org/abs/2410.10629), ICLR 2025
- [Radial Attention: $O(n \\log n)$ Sparse Attention with Energy Decay for Long Video Generation](https://github.com/mit-han-lab/radial-attention), ArXiv 2025

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

我们感谢 MIT-IBM Watson AI Lab、MIT 和 Amazon Science Hub、MIT AI Hardware Program、National Science Foundation、Packard Foundation、Dell、LG、现代和三星对本研究的支持。我们感谢 NVIDIA 捐赠的 DGX 服务器。

我们使用 [img2img-turbo](https://github.com/GaParmar/img2img-turbo) 训练草图到图像的 LoRA。我们的文本到图像和图像到图像 UI 分别基于 [playground-v.25](https://huggingface.co/spaces/playgroundai/playground-v2.5/blob/main/app.py) 和 [img2img-turbo](https://github.com/GaParmar/img2img-turbo/blob/main/gradio_sketch2image.py) 构建。我们的安全检查器来自 [hart](https://github.com/mit-han-lab/hart)。

Nunchaku 还受到许多开源库的启发，包括（但不限于）[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)、[vLLM](https://github.com/vllm-project/vllm)、[QServe](https://github.com/mit-han-lab/qserve)、[AWQ](https://github.com/mit-han-lab/llm-awq)、[FlashAttention-2](https://github.com/Dao-AILab/flash-attention) 和 [Atom](https://github.com/efeslab/Atom)。

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=mit-han-lab/nunchaku&type=Date)](https://www.star-history.com/#mit-han-lab/nunchaku&Date)
