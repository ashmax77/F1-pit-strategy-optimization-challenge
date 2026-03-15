#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "Scripts" / "python.exe"
TRAIN = REPO / "solution" / "train_model.py"
BENCH = REPO / "solution" / "benchmark_visible.py"

# Short representative sweep within requested ranges:
# epochs: 3-6, pairs: 80-140, lr: 0.02-0.04
COMBOS = [
    {"epochs": 3, "pairs": 80, "lr": 0.02},
    {"epochs": 3, "pairs": 110, "lr": 0.03},
    {"epochs": 4, "pairs": 100, "lr": 0.03},
    {"epochs": 5, "pairs": 110, "lr": 0.03},
    {"epochs": 6, "pairs": 80, "lr": 0.02},
    {"epochs": 6, "pairs": 140, "lr": 0.04},
]


def run_cmd(args):
    proc = subprocess.run(args, cwd=REPO, text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def run_one(combo):
    train_cmd = [
        str(PY), str(TRAIN),
        "--max-files", "30",
        "--epochs", str(combo["epochs"]),
        "--pairs-per-race", str(combo["pairs"]),
        "--learning-rate", str(combo["lr"]),
        "--l2", "1e-5",
    ]
    rc, out, err = run_cmd(train_cmd)
    if rc != 0:
        return {
            "combo": combo,
            "error": "train_failed",
            "train_tail": (out + "\n" + err)[-2000:],
        }

    rc, out, err = run_cmd([str(PY), str(BENCH)])
    if rc != 0:
        return {
            "combo": combo,
            "error": "bench_failed",
            "bench_tail": (out + "\n" + err)[-2000:],
        }

    metrics = json.loads(out)
    return {
        "combo": combo,
        "passed": metrics["passed"],
        "total": metrics["total"],
        "pass_rate": metrics["pass_rate"],
    }


def main():
    results = []
    for i, combo in enumerate(COMBOS, start=1):
        print(f"[{i}/{len(COMBOS)}] running {combo}")
        res = run_one(combo)
        print(json.dumps(res, indent=2))
        results.append(res)

    ok = [r for r in results if "pass_rate" in r]
    ok.sort(key=lambda r: (r["passed"], r["pass_rate"]), reverse=True)

    summary = {
        "results": results,
        "best": ok[0] if ok else None,
    }
    out_path = REPO / "solution" / "sweep_results.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", out_path)
    if ok:
        print("best", json.dumps(ok[0], indent=2))


if __name__ == "__main__":
    main()
