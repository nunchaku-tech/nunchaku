import subprocess
from pathlib import Path

import pytest

from nunchaku.utils import get_precision

EXAMPLES_DIR = Path("./examples/v1")

example_scripts = [str(f) for f in EXAMPLES_DIR.iterdir() if f.is_file() and f.suffix == ".py"]


@pytest.mark.parametrize("script_path", example_scripts)
def test_example_script_runs(script_path):
    if "sdxl" in script_path and get_precision() == "fp4":
        pytest.skip("Skip FP4 tests for SDXL!")
    result = subprocess.run(["python", script_path], text=True)
    print(f"Running {script_path} -> Return code: {result.returncode}")
    assert result.returncode == 0, f"{script_path} failed with code {result.returncode}"
