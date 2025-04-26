### ❗ 导入错误：`ImportError: cannot import name 'to_diffusers' from 'nunchaku.lora.flux' (...)`（例如 mit-han-lab/nunchaku#250）
此错误通常表示 `nunchaku` 库未正确安装。我们为 Windows 用户准备了分步安装指南：

📺 [英文教程](https://youtu.be/YHAVe-oM7U8?si=cM9zaby_aEHiFXk0) | 📺 [中文教程](https://www.bilibili.com/video/BV1BTocYjEk5/?share_source=copy_web&vd_source=8926212fef622f25cc95380515ac74ee) | 📖 [对应文本指南](https://github.com/mit-han-lab/nunchaku/blob/main/docs/setup_windows.md)

请同时检查以下常见原因：
* **您仅安装了 ComfyUI 插件（`ComfyUI-nunchaku`）而未安装核心 `nunchaku` 库。** 请遵循[README 中的安装说明](https://github.com/mit-han-lab/nunchaku?tab=readme-ov-file#installation)安装正确版本的 `nunchaku` 库。

* **您使用 `pip install nunchaku` 安装了错误包。**
  PyPI 上的 `nunchaku` 名称已被无关项目占用。请卸载错误包并按照[安装指南](https://github.com/mit-han-lab/nunchaku?tab=readme-ov-file#installation)操作。

* **（最常见）您正确安装了 `nunchaku`，但安装到了错误的 Python 环境中。**
  如果使用 ComfyUI 便携包，其 Python 解释器很可能不是系统默认版本。启动 ComfyUI 后，检查日志开头的 Python 路径，例如：
  ```text
  ** Python executable: G:\ComfyuI\python\python.exe
  ```
  使用以下命令安装到该环境：
  ```shell
  "G:\ComfyUI\python\python.exe" -m pip install <your-wheel-file>.whl
  ```
  示例（Python 3.11 + Torch 2.6）：
  ```shell
  "G:\ComfyUI\python\python.exe" -m pip install https://github.com/mit-han-lab/nunchaku/releases/download/v0.2.0/nunchaku-0.2.0+torch2.6-cp311-cp311-linux_x86_64.whl
  ```

* **您的工作目录中存在名为 `nunchaku` 的文件夹。**
  Python 可能会误加载本地文件夹而非已安装库。同时确保 `custom_nodes` 下的插件文件夹名为 `ComfyUI-nunchaku`，而非 `nunchaku`。

### ❗ 运行时错误：`Assertion failed: this->shape.dataExtent == other.shape.dataExtent, file ...Tensor.h`(例如 mit-han-lab/nunchaku#212)
此错误通常由使用与 GPU 不匹配的模型引起：
- 若使用 **Blackwell GPU（如 RTX 50 系列）**，请使用 **FP4** 模型。
- 其他 GPU 请使用 **INT4** 模型。

### ❗ 系统崩溃或蓝屏（例如 mit-han-lab/nunchaku#57）
我们观察到在使用 ComfyUI 时，图像生成后内存未正确释放可能导致系统不稳定或崩溃。我们正在积极调查此问题。若您有 ComfyUI 内存管理经验，欢迎协助！

### ❗ 内存不足或模型加载缓慢（例如 mit-han-lab/nunchaku#249、mit-han-lab/nunchaku#311、mit-han-lab/nunchaku#276）
尝试升级 CUDA 驱动，并设置环境变量 `NUNCHAKU_LOAD_METHOD` 为 `READ` 或 `READNOPIN`。

### ❗ 相同种子生成略微不同的图像（例如 mit-han-lab/nunchaku#229、mit-han-lab/nunchaku#294）
此现象由 GPU 计算顺序导致的微小精度噪声引起。强制固定计算顺序会显著降低性能，因此我们不计划调整此行为。

### ❓ PuLID 支持（例如 mit-han-lab/nunchaku#258）
PuLID 支持正在开发中，将在下一主要版本中加入。

### ~~❗ 断言错误：`Assertion failed: a.dtype() == b.dtype(), file ...misc_kernels.cu`（例如 mit-han-lab/nunchaku#30）~~
~~目前我们**仅支持 16 位版本的 [ControlNet-Union-Pro](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro)**。FP8 及其他 ControlNet 支持将在未来版本中加入。~~ ✅ 此问题已解决。

### ~~❗ Assertion Error：`assert image_rotary_emb.shape[2] == batch_size * (txt_tokens + img_tokens)`（例如 mit-han-lab/nunchaku#24）~~
~~当前**不支持推理时批量大小超过 1**。我们将在未来主要版本中支持此功能。~~ ✅ 自 [v0.3.0dev0](https://github.com/mit-han-lab/nunchaku/releases/tag/v0.3.0dev0) 起已支持多批量推理。
