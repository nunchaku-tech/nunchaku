# Contribution Guide

Welcome to **Nunchaku**! We appreciate your interest in contributing. This guide outlines how to set up your environment, run tests, and submit a Pull Request (PR). Whether you're fixing a minor bug or implementing a major feature, we encourage you to follow these steps for a smooth and efficient contribution process.

## 🚀 Setting Up & Building from Source

### 1. Fork and Clone the Repository

> 📌 **Note:** As a new contributor, you won’t have write access to the official Nunchaku repository. Please fork the repository to your own GitHub account, then clone your fork locally:

```shell
git clone https://github.com/<your_username>/nunchaku.git
```

### 2. Install Dependencies & Build

To install dependencies and build the project, follow the instructions in our [README](../README.md#installation).

## 🧹 Code Formatting with Pre-Commit

We use [pre-commit](https://pre-commit.com/) hooks to ensure code style consistency. Please install and run it before submitting your changes:

```shell
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

- `pre-commit run --all-files` manually triggers all checks and automatically fixes issues where possible. If it fails initially, re-run until all checks pass.

* ✅ **Ensure your code passes all checks before opening a PR.**

* 🚫 **Do not commit directly to the `main` branch.** Always create a feature branch (e.g., `feat/my-new-feature`), commit your changes there, and open a PR from that branch.

## 🧪 Running Unit Tests & Integrating with CI

Nunchaku uses `pytest` for unit testing. If you're adding a new feature, please include corresponding test cases in the [`tests`](../tests) directory.

For detailed guidance on testing, refer to the [`tests/README.md`](../tests/README.md).

## Acknowledgments

This contribution guide is adapted from [SGLang](https://docs.sglang.ai/references/contribution_guide.html). We thank them for the inspiration.
