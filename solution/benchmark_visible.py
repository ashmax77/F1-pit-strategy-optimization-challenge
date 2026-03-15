#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
inputs_dir = repo_root / "data" / "test_cases" / "inputs"
expected_dir = repo_root / "data" / "test_cases" / "expected_outputs"
simulator = repo_root / "solution" / "race_simulator.py"
python_exe = repo_root / ".venv" / "Scripts" / "python.exe"

passed = 0
total = 0
for i in range(1, 101):
    name = f"test_{i:03d}.json"
    inp = (inputs_dir / name).read_text(encoding="utf-8")
    expected = json.loads((expected_dir / name).read_text(encoding="utf-8"))["finishing_positions"]

    proc = subprocess.run(
        [str(python_exe), str(simulator)],
        input=inp,
        text=True,
        capture_output=True,
        check=True,
    )
    pred = json.loads(proc.stdout)["finishing_positions"]
    if pred == expected:
        passed += 1
    total += 1

print(json.dumps({"passed": passed, "total": total, "pass_rate": passed / total}, indent=2))
