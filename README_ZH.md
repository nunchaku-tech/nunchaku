<div align="center" id="nunchaku_logo">
  <img src="assets/nunchaku.svg" alt="logo" width="220"></img>
</div>
<h3 align="center">
<a href="http://arxiv.org/abs/2411.05007"><b>论文</b></a> | <a href="https://hanlab.mit.edu/projects/svdquant"><b>官网</b></a> | <a href="https://hanlab.mit.edu/blog/svdquant"><b>博客</b></a> | <a href="https://svdquant.mit.edu"><b>演示</b></a> | <a href="https://huggingface.co/collections/mit-han-lab/svdquant-67493c2c2e62a1fc6e93f45c"><b>HuggingFace</b></a> | <a href="https://modelscope.cn/collections/svdquant-468e8f780c2641"><b>ModelScope</b></a> | <a href="https://github.com/mit-han-lab/ComfyUI-nunchaku"><b>ComfyUI</b></a>
</h3>

<h3 align="center">
<a href="README.md"><b>English</b></a> | <a href="README_ZH.md"><b>中文</b></a>
</h3>

**Nunchaku**是一款专为4-bit神经网络优化的高性能推理引擎，基于我们的论文 [SVDQuant](http://arxiv.org/abs/2411.05007) 提出。底层量化库请参考 [DeepCompressor](https://github.com/mit-han-lab/deepcompressor)。

欢迎加入我们的用户群：[**Slack**](https://join.slack.com/t/nunchaku/shared_invite/zt-3170agzoz-NgZzWaTrEj~n2KEV3Hpl5Q)、[**Discord**](https://discord.gg/Wk6PnwX9Sm) 和 [**微信**](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/wechat.jpg)，与社区交流！更多详情请见[此处](https://github.com/mit-han-lab/nunchaku/issues/149)。如有任何问题、建议或贡献意向，欢迎随时联系！

## 最新动态

- **[2025-04-09]** 🎥 发布了[**英文**](https://youtu.be/YHAVe-oM7U8?si=cM9zaby_aEHiFXk0)和[**中文**](https://www.bilibili.com/video/BV1BTocYjEk5/?share_source=copy_web&vd_source=8926212fef622f25cc95380515ac74ee)教程视频，协助安装和使用Nunchaku。
- **[2025-04-09]** 📢 发布[四月开发路线图](https://github.com/mit-han-lab/nunchaku/issues/266)和[常见问题解答](https://github.com/mit-han-lab/nunchaku/discussions/262)，帮助社区快速上手并了解Nunchaku最新进展。
- **[2025-04-05]** 🚀 **Nunchaku v0.2.0 发布！** 支持[**多LoRA融合**](examples/flux.1-dev-multiple-lora.py)和[**ControlNet**](examples/flux.1-dev-controlnet-union-pro.py)，通过[**FP16 attention**](#fp16-attention)和[**First-Block Cache**](#first-block-cache)实现更快的推理速度。新增[**20系显卡支持**](examples/flux.1-dev-turing.py)，覆盖更多用户！
- **[2025-03-17]** 🚀 发布NVFP4 4-bit量化版[Shuttle-Jaguar](https://huggingface.co/mit-han-lab/svdq-int4-shuttle-jaguar)和FLUX.1工具集，升级INT4 FLUX.1工具模型。从[HuggingFace](https://huggingface.co/collections/mit-han-lab/svdquant-67493c2c2e62a1fc6e93f45c)或[ModelScope](https://modelscope.cn/collections/svdquant-468e8f780c2641)下载更新！
- **[2025-03-13]** 📦 ComfyUI节点[独立仓库](https://github.com/mit-han-lab/ComfyUI-nunchaku)发布，安装更便捷！节点版本v0.1.6上线，全面支持[4-bit Shuttle-Jaguar](https://huggingface.co/mit-han-lab/svdq-int4-shuttle-jaguar)！
- **[2025-03-07]** 🚀 **Nunchaku v0.1.4 发布！** 支持4-bit文本编码器和分层CPU offloading，FLUX最低显存需求降至**4 GiB**，同时保持**2–3倍加速**。修复分辨率、LoRA、内存锁定等稳定性问题，详情见更新日志！
- **[2025-02-20]** 🚀 发布[预编译wheel包](https://huggingface.co/mit-han-lab/nunchaku)，简化安装步骤！查看[安装指南](#安装指南)！
- **[2025-02-20]** 🚀 **NVIDIA RTX 5090支持NVFP4精度！** 相比INT4，NVFP4画质更优，在RTX 5090上比BF16快**约3倍**。[博客详解](https://hanlab.mit.edu/blog/svdquant-nvfp4)，[示例代码](./examples)及[在线演示](https://svdquant.mit.edu/flux1-schnell/)已上线！
- **[2025-02-18]** 🔥 新增[自定义LoRA转换](#自定义lora)和[模型量化](#自定义模型量化)指南！[ComfyUI](./comfyui)工作流支持**自定义LoRA**及**FLUX.1工具集**！
- **[2025-02-11]** 🎉 **[SVDQuant](http://arxiv.org/abs/2411.05007)入选ICLR 2025 Spotlight！FLUX.1工具集使用演示上线！** [使用演示](#使用演示)已更新！[深度图生成演示](https://svdquant.mit.edu/flux1-depth-dev/)同步开放！

<details>
<summary>更多动态</summary>

- **[2025-02-04]** **🚀 4-bit量化版[FLUX.1工具集](https://blackforestlabs.ai/flux-1-tools/)发布！** 相比原模型提速**2-3倍**。[示例代码](./examples)已更新，**ComfyUI支持即将到来！**
- **[2025-01-23]** 🚀 **支持4-bit量化[SANA](https://nvlabs.github.io/Sana/)！** 相比16位模型提速2-3倍。[使用示例](./examples/sana_1600m_pag.py)和[部署指南](app/sana/t2i)已发布，体验[在线演示](https://svdquant.mit.edu)！
- **[2025-01-22]** 🎉 [**SVDQuant**](http://arxiv.org/abs/2411.05007) 被 **ICLR 2025** 接收！
- **[2024-12-08]** 支持 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)，详情见 [mit-han-lab/ComfyUI-nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku)。
- **[2024-11-07]** 🔥 最新 **W4A4** 扩散模型量化工作 [**SVDQuant**](https://hanlab.mit.edu/projects/svdquant) 开源！量化库 [**DeepCompressor**](https://github.com/mit-han-lab/deepcompressor) 同步发布。

</details>

## 项目概览

![teaser](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/teaser.jpg)
SVDQuant 是一种支持4-bit权重和激活的后训练量化技术，能有效保持视觉质量。在12B FLUX.1-dev模型上，相比BF16模型实现了3.6倍内存压缩。通过消除CPU offloading，在16GB笔记本RTX 4090上比16位模型快8.7倍，比NF4 W4A16基线快3倍。在PixArt-∑模型上，其视觉质量显著优于其他W4A4甚至W4A8方案。"E2E"表示包含文本编码器和VAE解码器的端到端延迟。

**SVDQuant: 通过低秩分量吸收异常值实现4-bit扩散模型量化**<br>
[Muyang Li](https://lmxyy.me)\*, [Yujun Lin](https://yujunlin.com)\*, [Zhekai Zhang](https://hanlab.mit.edu/team/zhekai-zhang)\*, [Tianle Cai](https://www.tianle.website/#/), [Xiuyu Li](https://xiuyuli.com), [Junxian Guo](https://github.com/JerryGJX), [Enze Xie](https://xieenze.github.io), [Chenlin Meng](https://cs.stanford.edu/~chenlin/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Song Han](https://hanlab.mit.edu/songhan) <br>
*麻省理工学院、英伟达、卡内基梅隆大学、普林斯顿大学、加州大学伯克利分校、上海交通大学、pika实验室* <br>

https://github.com/user-attachments/assets/fdd4ab68-6489-4c65-8768-259bd866e8f8

## 方法原理

#### 量化方法 -- SVDQuant

![intuition](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/intuition.gif)SVDQuant三阶段示意图。阶段1：原始激活 $\boldsymbol{X}$ 和权重 $\boldsymbol{W}$ 均含异常值，4-bit量化困难。阶段2：将激活异常值迁移至权重，得到更易量化的激活 $\hat{\boldsymbol{X}}$ 和更难量化的权重 $\hat{\boldsymbol{W}}$ 。阶段3：通过SVD将 $\hat{\boldsymbol{W}}$ 分解为低秩分量 $\boldsymbol{L}_1\boldsymbol{L}_2$ 和残差 $\hat{\boldsymbol{W}}-\boldsymbol{L}_1\boldsymbol{L}_2$ ，低秩分支以16位精度运行缓解量化难度。

#### Nunchaku引擎设计

![engine](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/engine.jpg) (a) 原始低秩分支（秩32）因额外读写16位数据引入57%的延迟。Nunchaku通过核融合优化。(b) 将下投影与量化、上投影与4-bit计算分别融合，减少数据搬运。

## 性能表现

![efficiency](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/efficiency.jpg)SVDQuant 将12B FLUX.1模型的体积压缩了3.6倍，同时将原始16位模型的显存占用减少了3.5倍。借助Nunchaku，我们的INT4模型在桌面和笔记本的NVIDIA RTX 4090 GPU上比NF4 W4A16基线快了3.0倍。值得一提的是，在笔记本4090上，通过消除CPU offloading，总体加速达到了10.1倍。我们的NVFP4模型在RTX 5090 GPU上也比BF16和NF4快了3.1倍。

## 安装指南

我们提供了在 Windows 上安装和使用 Nunchaku 的教学视频，支持[**英文**](https://youtu.be/YHAVe-oM7U8?si=cM9zaby_aEHiFXk0)和[**中文**](https://www.bilibili.com/video/BV1BTocYjEk5/?share_source=copy_web&vd_source=8926212fef622f25cc95380515ac74ee)两个版本。同时，你也可以参考对应的图文教程 [`docs/setup_windows.md`](docs/setup_windows.md)。如果在安装过程中遇到问题，建议优先查阅这些资源。

### Wheel包安装

#### 前置条件
确保已安装 [PyTorch>=2.5](https://pytorch.org/)。例如：

```shell
pip install torch==2.6 torchvision==0.21 torchaudio==2.6
```

#### 安装nunchaku
从[Hugging Face](https://huggingface.co/mit-han-lab/nunchaku/tree/main)、[ModelScope](https://modelscope.cn/models/Lmxyy1999/nunchaku)或[GitHub release](https://github.com/mit-han-lab/nunchaku/releases)选择对应Python和PyTorch版本的wheel。例如Python 3.11和PyTorch 2.6：

```shell
pip install https://huggingface.co/mit-han-lab/nunchaku/resolve/main/nunchaku-0.2.0+torch2.6-cp311-cp311-linux_x86_64.whl
```

##### ComfyUI用户

若使用**ComfyUI便携包**，请确保将`nunchaku`安装到ComfyUI自带的Python环境。查看ComfyUI日志获取Python路径：

```text
** Python executable: G:\ComfyuI\python\python.exe
```

使用该Python安装wheel：

```shell
"G:\ComfyUI\python\python.exe" -m pip install <your-wheel-file>.whl
```

**示例**：为Python 3.11和PyTorch 2.6安装：

```shell
"G:\ComfyUI\python\python.exe" -m pip install https://github.com/mit-han-lab/nunchaku/releases/download/v0.2.0/nunchaku-0.2.0+torch2.6-cp311-cp311-linux_x86_64.whl
```

##### Blackwell显卡用户（50系列）

若使用Blackwell显卡（如50系列），请安装PyTorch 2.7及以上版本，并使用**FP4模型**。

### 源码编译

**注意**：

* Linux需CUDA≥12.2，Windows需CUDA≥12.6。Blackwell显卡需CUDA≥12.8。
* Windows用户请参考[此问题](https://github.com/mit-han-lab/nunchaku/issues/6)升级MSVC编译器。
* 支持SM_75（Turing：RTX 2080）、SM_86（Ampere：RTX 3090）、SM_89（Ada：RTX 4090）、SM_80（A100）架构显卡，详见[此问题](https://github.com/mit-han-lab/nunchaku/issues/1)。

1. 安装依赖：

   ```shell
   conda create -n nunchaku python=3.11
   conda activate nunchaku
   pip install torch torchvision torchaudio
   pip install ninja wheel diffusers transformers accelerate sentencepiece protobuf huggingface_hub

   # Gradio演示依赖
   pip install peft opencv-python gradio spaces GPUtil
   ```

   Blackwell用户需安装PyTorch nightly（CUDA 12.8）：

   ```shell
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
   ```

2. 编译安装：
   确保`gcc/g++≥11`。Linux用户可通过Conda安装：

    ```shell
    conda install -c conda-forge gxx=11 gcc=11
    ```

    Windows用户请安装最新[Visual Studio](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2030&passive=false)。

    编译命令：

    ```shell
    git clone https://github.com/mit-han-lab/nunchaku.git
    cd nunchaku
    git submodule init
    git submodule update
    python setup.py develop
    ```

    打包wheel：

    ```shell
    NUNCHAKU_INSTALL_MODE=ALL NUNCHAKU_BUILD_WHEELS=1 python -m build --wheel --no-isolation
    ```

    设置`NUNCHAKU_INSTALL_MODE=ALL`确保wheel支持所有显卡架构。

## 使用示例

在[示例](examples)中，我们提供了运行4-bit[FLUX.1](https://github.com/black-forest-labs/flux)和[SANA](https://github.com/NVlabs/Sana)模型的极简脚本，API与[diffusers](https://github.com/huggingface/diffusers)兼容。例如[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)脚本：

```python
import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

precision = get_precision()  # 自动检测GPU支持的精度（int4或fp4）
transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-dev")
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")
image = pipeline("举着'Hello World'标牌的猫咪", num_inference_steps=50, guidance_scale=3.5).images[0]
image.save(f"flux.1-dev-{precision}.png")
```

**注意**：**Turing显卡用户（如20系列）**需设置`torch_dtype=torch.float16`并使用`nunchaku-fp16`注意力模块，完整示例见[`examples/flux.1-dev-turing.py`](examples/flux.1-dev-turing.py)。

### FP16 Attention

除FlashAttention-2外，Nunchaku提供定制FP16 Attention实现，在30/40/50系显卡上提速**1.2倍**且无损精度。启用方式：

```python
transformer.set_attention_impl("nunchaku-fp16")
```

完整示例见[`examples/flux.1-dev-fp16attn.py`](examples/flux.1-dev-fp16attn.py)。

### First-Block Cache

Nunchaku支持[First-Block Cache](https://github.com/chengzeyi/ParaAttention?tab=readme-ov-file#first-block-cache-our-dynamic-caching)加速长步去噪。启用方式：

```python
apply_cache_on_pipe(pipeline, residual_diff_threshold=0.12)
```

`residual_diff_threshold`越大速度越快但可能影响质量，推荐值`0.12`，50步推理提速2倍，30步提速1.4倍。完整示例见[`examples/flux.1-dev-cache.py`](examples/flux.1-dev-cache.py)。

### CPU offloading

最小化显存占用至**4 GiB**，设置`offload=True`并启用CPU offloading：

```python
pipeline.enable_sequential_cpu_offload()
```

完整示例见[`examples/flux.1-dev-offload.py`](examples/flux.1-dev-offload.py)。

## 自定义LoRA

![lora](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/lora.jpg)

[SVDQuant](http://arxiv.org/abs/2411.05007) 可以无缝集成现有的 LoRA，而无需重新量化。你可以简单地通过以下方式使用你的 LoRA：

```python
transformer.update_lora_params(path_to_your_lora)
transformer.set_lora_strength(lora_strength)
```

`path_to_your_lora` 也可以是一个远程的 HuggingFace 路径。在 [`examples/flux.1-dev-lora.py`](examples/flux.1-dev-lora.py) 中，我们提供了一个运行 [Ghibsky](https://huggingface.co/aleksa-codes/flux-ghibsky-illustration) LoRA 的最小示例脚本，结合了 SVDQuant 的 4-bit FLUX.1-dev：

```python
import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

precision = get_precision()  # 自动检测你的精度是 'int4' 还是 'fp4'，取决于你的 GPU
transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-dev")
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

### LoRA 相关代码 ###
transformer.update_lora_params(
    "aleksa-codes/flux-ghibsky-illustration/lora.safetensors"
)  # 你的 LoRA safetensors 路径，也可以是远程 HuggingFace 路径
transformer.set_lora_strength(1)  # 在这里设置你的 LoRA 强度
### LoRA 相关代码结束 ###

image = pipeline(
    "GHIBSKY 风格，被雪覆盖的舒适山间小屋，烟囱里冒出袅袅炊烟，窗户透出温暖诱人的灯光",  # noqa: E501
    num_inference_steps=25,
    guidance_scale=3.5,
).images[0]
image.save(f"flux.1-dev-ghibsky-{precision}.png")
```

如果需要组合多个 LoRA，可以使用 `nunchaku.lora.flux.compose.compose_lora` 来实现组合。用法如下：

```python
composed_lora = compose_lora(
    [
        ("PATH_OR_STATE_DICT_OF_LORA1", lora_strength1),
        ("PATH_OR_STATE_DICT_OF_LORA2", lora_strength2),
        # 根据需要添加更多 LoRA
    ]
)  # 在使用组合 LoRA 时在此处设置每个 LoRA 的强度
transformer.update_lora_params(composed_lora)
```

你可以为列表中的每个 LoRA 指定单独的强度。完整的示例请参考 [`examples/flux.1-dev-multiple-lora.py`](examples/flux.1-dev-multiple-lora.py)。

**对于 ComfyUI 用户，你可以直接使用我们的 LoRA 加载器。转换后的 LoRA 已被弃用，请参考 [mit-han-lab/ComfyUI-nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku) 获取更多详细信息。**

## ControlNets

Nunchaku 支持 [FLUX.1-tools](https://blackforestlabs.ai/flux-1-tools/) 和 [FLUX.1-dev-ControlNet-Union-Pro](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro) 模型。示例脚本可以在 [`examples`](examples) 目录中找到。

![control](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/control.jpg)

## ComfyUI

请参考 [mit-han-lab/ComfyUI-nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku) 获取在 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 中的使用方法。

## 使用演示

* FLUX.1 模型
  * 文生图：见 [`app/flux.1/t2i`](app/flux.1/t2i)。
  * 草图生成图像 ([pix2pix-Turbo](https://github.com/GaParmar/img2img-turbo))：见 [`app/flux.1/sketch`](app/flux.1/sketch)。
  * 深度/Canny 边缘生成图像 ([FLUX.1-tools](https://blackforestlabs.ai/flux-1-tools/))：见 [`app/flux.1/depth_canny`](app/flux.1/depth_canny)。
  * 修复 ([FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev))：见 [`app/flux.1/fill`](app/flux.1/fill)。
  * Redux ([FLUX.1-Redux-dev](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev))：见 [`app/flux.1/redux`](app/flux.1/redux)。
* SANA：
  * 文生图：见 [`app/sana/t2i`](app/sana/t2i)。

## 自定义模型量化

请参考 [mit-han-lab/deepcompressor](https://github.com/mit-han-lab/deepcompressor/tree/main/examples/diffusion)。更简单的流程即将推出。

## 基准测试

请参考 [app/flux/t2i/README.md](app/flux/t2i/README.md) 获取重现我们论文质量结果和对 FLUX.1 模型进行推理延迟基准测试的说明。

## 路线图

请查看 [此处](https://github.com/mit-han-lab/nunchaku/issues/266) 获取四月的路线图。

## 贡献
我们诚挚欢迎社区贡献！请参阅[贡献指南](docs/contribution_guide_ZH.md)了解如何为 Nunchaku 贡献代码。

## 问题排查

使用 Nunchaku 时遇到问题？请先查阅我们的[常见问题解答](docs/faq_ZH.md)寻找解决方案。若仍需要帮助，可通过[open an issue](https://github.com/mit-han-lab/nunchaku/issues)联系我们。也欢迎您通过 [**Slack**](https://join.slack.com/t/nunchaku/shared_invite/zt-3170agzoz-NgZzWaTrEj~n2KEV3Hpl5Q)、[**Discord**](https://discord.gg/Wk6PnwX9Sm) 或 [**微信**](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/wechat.jpg) 加入我们的社区讨论。

## 联系我们

有意采用 SVDQuant 或 Nunchaku 的企业，包括技术咨询、赞助机会或合作咨询，请联系 muyangli@mit.edu。

## 相关项目

* [Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models](https://arxiv.org/abs/2211.02048), NeurIPS 2022 & T-PAMI 2023
* [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438), ICML 2023
* [Q-Diffusion: Quantizing Diffusion Models](https://arxiv.org/abs/2302.04304), ICCV 2023
* [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978), MLSys 2024
* [DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models](https://arxiv.org/abs/2402.19481), CVPR 2024
* [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/abs/2405.04532), MLSys 2025
* [SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers](https://arxiv.org/abs/2410.10629), ICLR 2025

## 引用

如果你觉得 `nunchaku` 对你的研究有用或相关，请引用我们的论文：

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

感谢 MIT-IBM Watson AI Lab、MIT 和Amazon Science Hub、MIT AI Hardware Program、National Science Foundation、Packard Foundation、Dell、LG、Hyundai和Samsung对本研究的支持。感谢 NVIDIA 捐赠 DGX 服务器。

我们使用 [img2img-turbo](https://github.com/GaParmar/img2img-turbo) 训练草图生成图像的 LoRA。我们的文生图和图像生成用户界面基于 [playground-v.25](https://huggingface.co/spaces/playgroundai/playground-v2.5/blob/main/app.py) 和 [img2img-turbo](https://github.com/GaParmar/img2img-turbo/blob/main/gradio_sketch2image.py) 构建。我们的安全检查器来自 [hart](https://github.com/mit-han-lab/hart)。

Nunchaku 还受到许多开源库的启发，包括（但不限于）[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)、[vLLM](https://github.com/vllm-project/vllm)、[QServe](https://github.com/mit-han-lab/qserve)、[AWQ](https://github.com/mit-han-lab/llm-awq)、[FlashAttention-2](https://github.com/Dao-AILab/flash-attention) 和 [Atom](https://github.com/efeslab/Atom)。
