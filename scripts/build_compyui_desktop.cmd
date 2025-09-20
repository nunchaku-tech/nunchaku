@echo off
setlocal

REM ==============================
REM  ComfyUI Desktop Build Script
REM ==============================

set PYTHON_VERSION=3.12
set TORCH_VERSION=2.7
REM Node.js version must be complete version string instead of 20 or 20.19
set NODE_VERSION=20.18.0
set YARN_VERSION=4.5.0
set NUNCHAKU_VERSION=1.0.0
set TORCHAUDIO_VERSION=2.7
set TORCHVISION_VERSION=0.22
set CUDA_PIP_INDEX=cu128

set PYTHON_VERSION_STR=%PYTHON_VERSION:.=%

set WORK_DIR=%cd%

REM 1. Install Python 3.12 silently with winget
echo Installing Python %PYTHON_VERSION%...
winget install -e --id Python.Python.%PYTHON_VERSION% --accept-source-agreements --accept-package-agreements -h
if %errorlevel% neq 0 (
    echo Failed to install Python %PYTHON_VERSION%
    exit /b 1
)

REM Assume Python 3.12 installs here:
set PYTHON_EXE="%LocalAppData%\Programs\Python\Python%PYTHON_VERSION_STR%\python.exe"

REM 2. Install uv package
echo Installing uv package...
%PYTHON_EXE% -m pip install --upgrade pip
%PYTHON_EXE% -m pip install uv

REM 3. Install NVM for Windows (if not already installed)
echo Installing NVM for Windows...
winget install -e --id CoreyButler.NVMforWindows --accept-source-agreements --accept-package-agreements -h
if %errorlevel% neq 0 (
    echo Failed to install NVM
    exit /b 1
)

REM 4. Install Node.js 20 via NVM
echo Installing Node.js %NODE_VERSION% with NVM...
set NVM_HOME=%LocalAppData%\nvm
cd %NVM_HOME%
nvm install %NODE_VERSION%
nvm use %NODE_VERSION%

REM 5. Install Yarn using npm
REM Note: this step needs admin permission
echo Installing yarn...
npm install -g yarn
corepack enable
yarn use %YARN_VERSION%

REM 6. Clone ComfyUI desktop repo
echo Cloning ComfyUI Desktop...
cd %WORK_DIR%
git clone https://github.com/Comfy-Org/desktop.git
cd desktop
git clone https://github.com/nunchaku-tech/ComfyUI-nunchaku.git assets/ComfyUI/custom_nodes/ComfyUI-nunchaku

REM 7. Install node modules and rebuild electron
echo Rebuilding native modules...
yarn install
npx --yes electron-rebuild
yarn make:assets

REM 8. Overwrite override.txt with torch 2.7 + custom nunchaku wheel
echo Writing override.txt...

set NUNCHAKU_URL=https://github.com/nunchaku-tech/nunchaku/releases/download/v%NUNCHAKU_VERSION%/nunchaku-%NUNCHAKU_VERSION%+torch%TORCH_VERSION%-cp%PYTHON_VERSION_STR%-cp%PYTHON_VERSION_STR%-win_amd64.whl

(
echo torch==%TORCH_VERSION%+%CUDA_PIP_INDEX%
echo torchaudio==%TORCHAUDIO_VERSION%+%CUDA_PIP_INDEX%
echo torchvision==%TORCHVISION_VERSION%+%CUDA_PIP_INDEX%
echo nunchaku @ %NUNCHAKU_URL%
) > assets\override.txt
echo nunchaku >> assets\ComfyUI\requirements.txt

REM 9. Build compiled requirements with uv
echo Rebuilding requirements (windows_nvidia.compiled)...
assets\uv\win\uv.exe pip compile assets\ComfyUI\requirements.txt assets\ComfyUI\custom_nodes\ComfyUI-Manager\requirements.txt ^
--emit-index-annotation --emit-index-url --index-strategy unsafe-best-match ^
-o assets\requirements\windows_nvidia.compiled ^
--override assets\override.txt ^
--index-url https://pypi.org/simple ^
--extra-index-url https://download.pytorch.org/whl/%CUDA_PIP_INDEX%

REM 10. Build for NVIDIA users on Windows
echo Building ComfyUI for NVIDIA...
yarn make:nvidia

echo ==========================================
echo ✅ Build process completed successfully!
echo ==========================================

endlocal
