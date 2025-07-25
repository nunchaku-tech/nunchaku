import subprocess
from pathlib import Path


def run_all_tests():
    test_dir = Path(__file__).parent.parent / "tests"
    test_files = []
    for file_path in test_dir.rglob("test_*.py"):
        # Ignore tests/flux/test_flux_examples.py
        try:
            rel_path = file_path.relative_to(test_dir)
        except ValueError:
            continue
        if str(rel_path) == "flux/test_flux_examples.py":
            continue
        test_files.append(str(file_path))
    print("Running tests:")
    for test_file in test_files:
        print(f"  {test_file}")

    failed_tests = []
    for test_file in test_files:
        print(f"Running {test_file} ...")
        result = subprocess.run(["pytest", test_file])
        if result.returncode != 0:
            print(f"Test failed: {test_file}")
            failed_tests.append(test_file)
        else:
            print(f"Test passed: {test_file}")

    if failed_tests:
        print("Some tests failed.")
        for test_file in failed_tests:
            print(f"  {test_file}")
        exit(1)
    else:
        print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()
