@echo off
setlocal enabledelayedexpansion

REM Define Python and Torch versions
set "python_versions=3.10 3.11 3.12"
set "torch_versions=2.5 2.6"
set "cuda_version=12.4"

REM Iterate over Python and Torch versions
for %%P in (%python_versions%) do (
    for %%T in (%torch_versions%) do (
        REM Python 3.13 only supports Torch 2.6 and above
        if not "%%P"=="3.13" (
            echo Building with Python %%P, Torch %%T, CUDA %cuda_version%...
            call scripts\build_windows_wheel.cmd %%P %%T %cuda_version%
        ) else if not "%%T"=="2.5" (
            echo Building with Python %%P, Torch %%T, CUDA %cuda_version%...
            call scripts\build_windows_wheel.cmd %%P %%T %cuda_version%
        )
    )
)

call scripts\build_windows_wheel.cmd 3.10 2.7 12.8
call scripts\build_windows_wheel.cmd 3.11 2.7 12.8
call scripts\build_windows_wheel.cmd 3.12 2.7 12.8

REM call scripts\build_windows_wheel_cu128.cmd 3.10 2.8 12.8
REM call scripts\build_windows_wheel_cu128.cmd 3.11 2.8 12.8
REM call scripts\build_windows_wheel_cu128.cmd 3.12 2.8 12.8

echo All builds completed successfully!
exit /b 0
