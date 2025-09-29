import subprocess
from pathlib import Path

from tqdm import tqdm


def run_all_tests():
    test_dir = Path("tests")
    test_files = []
    for file_path in test_dir.rglob("test_*.py"):
        # Ignore tests/flux/test_flux_examples.py
        try:
            rel_path = file_path.relative_to(test_dir)
        except ValueError:
            continue
        if "test_flux_examples.py" in str(rel_path):
            continue
        if "test_examples.py" in str(rel_path):
            continue
        test_files.append(str(file_path))
    print("Running tests:")
    for test_file in sorted(test_files):
        print(f"  {test_file}")

    failed_tests = []
    for test_file in tqdm(test_files):
        print(f"Running {test_file} ...")
        result = subprocess.run(["pytest", "--reruns", "2", "--reruns-delay", "0", "-vv", "-x", test_file])
        if result.returncode != 0:
            print(f"Test failed: {test_file}")
            failed_tests.append(test_file)
        else:
            print(f"Test passed: {test_file}")
        if len(failed_tests) > 0:
            break

    if failed_tests:
        print("Some tests failed.")
        for test_file in failed_tests:
            print(f"  {test_file}")
        exit(1)
    else:
        print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()
