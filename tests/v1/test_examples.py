import os
import subprocess
from pathlib import Path

import pytest

from nunchaku.utils import get_precision

EXAMPLES_DIR = Path("./examples/v1")

example_scripts = [str(f) for f in EXAMPLES_DIR.iterdir() if f.is_file() and f.suffix == ".py"]


@pytest.mark.parametrize("script_name", example_scripts)
def test_example_script_runs(script_name):
    if "sdxl" in script_name and get_precision() == "fp4":
        pytest.skip("Skip FP4 tests for SDXL!")
    script_path = os.path.join(EXAMPLES_DIR, script_name)
    result = subprocess.run(["python", script_path], text=True)
    print(f"Running {script_path} -> Return code: {result.returncode}")
    assert result.returncode == 0, f"{script_path} failed with code {result.returncode}"
