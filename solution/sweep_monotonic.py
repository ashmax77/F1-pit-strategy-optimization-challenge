#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "Scripts" / "python.exe"
TRAIN = REPO / "solution" / "train_model.py"
BENCH = REPO / "solution" / "benchmark_visible.py"
MODEL = REPO / "solution" / "model_params.json"

# Fixed best fine-tuning settings from prior sweep
EPOCHS = 6
PAIRS = 140
LR = 0.04
L2 = 1e-5

STRENGTHS = [0.0, 0.08, 0.12, 0.18]


def run_cmd(args):
    proc = subprocess.run(args, cwd=REPO, text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def run_one(strength):
    train_cmd = [
        str(PY), str(TRAIN),
        "--max-files", "30",
        "--epochs", str(EPOCHS),
        "--pairs-per-race", str(PAIRS),
        "--learning-rate", str(LR),
        "--l2", str(L2),
        "--monotonic-strength", str(strength),
    ]
    rc, out, err = run_cmd(train_cmd)
    if rc != 0:
        return {
            "monotonic_strength": strength,
            "error": "train_failed",
            "tail": (out + "\n" + err)[-2500:],
        }

    # Keep a copy of each trained model for comparison
    copy_path = REPO / "solution" / f"model_params_mono_{str(strength).replace('.', '_')}.json"
    copy_path.write_text(MODEL.read_text(encoding="utf-8"), encoding="utf-8")

    rc, out, err = run_cmd([str(PY), str(BENCH)])
    if rc != 0:
        return {
            "monotonic_strength": strength,
            "error": "bench_failed",
            "tail": (out + "\n" + err)[-2500:],
        }

    metrics = json.loads(out)
    return {
        "monotonic_strength": strength,
        "passed": metrics["passed"],
        "total": metrics["total"],
        "pass_rate": metrics["pass_rate"],
        "model_copy": str(copy_path.relative_to(REPO)).replace('\\', '/'),
    }


def main():
    results = []
    for i, strength in enumerate(STRENGTHS, start=1):
        print(f"[{i}/{len(STRENGTHS)}] monotonic_strength={strength}")
        res = run_one(strength)
        print(json.dumps(res, indent=2))
        results.append(res)

    ok = [r for r in results if "pass_rate" in r]
    ok.sort(key=lambda r: (r["passed"], r["pass_rate"]), reverse=True)

    summary = {
        "fixed_hparams": {
            "epochs": EPOCHS,
            "pairs_per_race": PAIRS,
            "learning_rate": LR,
            "l2": L2,
        },
        "results": results,
        "best": ok[0] if ok else None,
    }
    out_path = REPO / "solution" / "sweep_monotonic_results.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", out_path)

    if ok:
        best = ok[0]
        # Restore best model as active model_params.json
        best_model_path = REPO / best["model_copy"]
        MODEL.write_text(best_model_path.read_text(encoding="utf-8"), encoding="utf-8")
        print("restored_best_model", best["model_copy"])
        print("best", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
