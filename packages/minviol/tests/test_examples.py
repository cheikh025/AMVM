"""The shipped examples have to run, on small settings."""
import pathlib
import subprocess
import sys

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("script,args", [
    ("tomography.py", ["--size", "16", "--angles", "8", "--levels", "2",
                       "--seconds", "1", "--device", "cpu"]),
    ("feasibility.py", ["--constraints", "2000", "--variables", "32",
                        "--seconds", "1", "--device", "cpu"]),
])
def test_example_runs(script, args):
    finished = subprocess.run([sys.executable, str(EXAMPLES / script), *args],
                              capture_output=True, text=True, timeout=180)
    assert finished.returncode == 0, finished.stderr
    assert "violation recomputed from scratch" in finished.stdout
