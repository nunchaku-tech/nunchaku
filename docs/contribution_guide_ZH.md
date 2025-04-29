# 贡献指南

欢迎来到 **Nunchaku**！我们非常感谢您的贡献兴趣。本指南将指导您完成环境配置、测试运行和提交拉取请求（PR）的流程。无论您是修复小问题还是开发重要功能，都请遵循以下步骤以确保贡献过程顺畅高效。

## 🚀 环境设置与源码构建

### 1. Fork并克隆仓库

> 📌 **注意**：作为新贡献者，您暂时没有官方仓库的写入权限。请先将仓库Fork到自己的GitHub账号，然后克隆到本地：

```shell
git clone https://github.com/<your_username>/nunchaku.git
```

### 2. 安装依赖与构建

安装依赖并构建项目的具体步骤请参考[README](../README.md#installation)中的说明。

## 🧹 使用Pre-Commit进行代码格式化

我们通过[pre-commit](https://pre-commit.com/)hooks确保代码风格统一。提交更改前请务必安装并运行：

```shell
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

- `pre-commit run --all-files` 会手动触发所有检查并自动修复可解决的问题。若初次运行失败，请重复执行直至全部通过

* ✅ **提交PR前请确保代码通过所有检查**

* 🚫 **禁止直接提交到`main`分支**。请始终创建功能分支（如`feat/my-new-feature`），并在该分支上提交更改后发起PR

## 🧪 单元测试与CI集成

Nunchaku使用`pytest`进行单元测试。新增功能时，请在[`tests`](../tests)目录中添加对应的测试用例。

更多测试细节请参考[`tests/README.md`](../tests/README.md)。

## 致谢

本贡献指南改编自[SGLang](https://docs.sglang.ai/references/contribution_guide.html)，感谢他们的灵感启发。
