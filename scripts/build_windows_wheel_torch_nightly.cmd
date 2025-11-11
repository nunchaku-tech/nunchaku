@echo off
setlocal enabledelayedexpansion

:: get arguments
set PYTHON_VERSION=%1
set TORCH_VERSION=%2
set CUDA_VERSION=%3

set CUDA_SHORT_VERSION=%CUDA_VERSION:.=%
echo %CUDA_SHORT_VERSION%

:: conda environment name
set ENV_NAME=build_env_%PYTHON_VERSION%_%TORCH_VERSION%
echo Using conda environment: %ENV_NAME%

:: create conda environment
call conda create -y -n %ENV_NAME% python=%PYTHON_VERSION%
call conda activate %ENV_NAME%

:: install dependencies
call pip install ninja setuptools wheel build

call pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu%CUDA_SHORT_VERSION%

:: set environment variables
set NUNCHAKU_INSTALL_MODE=ALL
set NUNCHAKU_BUILD_WHEELS=1

:: cd to the parent directory
cd /d "%~dp0.."
if exist build rd /s /q build

:: set up Visual Studio compilation environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -startdir=none -arch=x64 -host_arch=x64
set DISTUTILS_USE_SDK=1

:: build wheels
python -m build --wheel --no-isolation

:: exit conda
call conda deactivate
call conda remove -y -n %ENV_NAME% --all

echo Build complete!
