#!/usr/bin/env python3
"""Private Kaggle T4 launcher for the 2.13M-parameter real-data RSNN Code V2 experiment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SOURCE_COMMIT = "__SOURCE_COMMIT__"
REPO = "https://github.com/msabz/Al.git"
BRANCH = "feat/open-growth-rsnn-python-code-v2"
WORK = Path("/kaggle/working")
SRC = WORK / "rsnn_code_v2_src"
EXPORT = WORK / "rsnn-code-v2-output"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    started = time.time()
    if not SOURCE_COMMIT or SOURCE_COMMIT.startswith("__"):
        raise SystemExit("SOURCE_COMMIT placeholder was not replaced")

    print("GPU_CHECK", flush=True)
    import torch
    print("torch=", torch.__version__, "cuda=", torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/T4 is required for Code V2")
    print("gpu=", torch.cuda.get_device_name(0), flush=True)

    shutil.rmtree(SRC, ignore_errors=True)
    shutil.rmtree(EXPORT, ignore_errors=True)
    SRC.mkdir(parents=True)
    EXPORT.mkdir(parents=True)

    run(["git", "init"], SRC)
    run(["git", "remote", "add", "origin", REPO], SRC)
    run(["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT], SRC)
    run(["git", "checkout", "--detach", "FETCH_HEAD"], SRC)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SRC, text=True).strip()
    if head != SOURCE_COMMIT:
        raise SystemExit(f"source mismatch: {head} != {SOURCE_COMMIT}")

    run([sys.executable, "-m", "py_compile",
         "code_model/train_rsnn_code_v1.py", "code_model/run_integer_only_v2.py",
         "code_model/collect_real_python.py", "code_model/evaluate_generation_v2.py"], SRC)

    run([sys.executable, "code_model/collect_real_python.py",
         "--output-dir", "code_model/data", "--max-per-repo", "2200"], SRC)

    # 2,128,963 parameters: 259x384 embedding + three 576-unit recurrent LIF layers + 259-way head.
    run([sys.executable, "-m", "code_model.run_integer_only_v2",
         "--train", "code_model/data/train.jsonl",
         "--valid", "code_model/data/valid.jsonl",
         "--out-dir", "code_model/output",
         "--steps", "1200",
         "--batch-size", "4",
         "--seq-len", "192",
         "--emb-dim", "384",
         "--hidden", "576",
         "--layers", "3",
         "--initial-sparsity", "0.75",
         "--lr", "0.0015",
         "--structural-interval", "150",
         "--seed", "29"], SRC)

    run([sys.executable, "code_model/evaluate_generation_v2.py",
         "--checkpoint", "code_model/output/checkpoint_latest.pt",
         "--output", "code_model/output/generation_report.json",
         "--max-new", "220"], SRC)

    report = json.loads((SRC / "code_model/output/report.json").read_text())
    gen = json.loads((SRC / "code_model/output/generation_report.json").read_text())
    provenance = json.loads((SRC / "code_model/data/provenance.json").read_text())

    if int(report["total_parameters"]) < 2_000_000:
        raise SystemExit(f"model smaller than requested experiment: {report['total_parameters']}")
    if provenance["train"]["examples"] < 5000 or provenance["valid"]["examples"] < 1000:
        raise SystemExit("real dataset gate failed")
    if not report.get("full_int8_target"):
        raise SystemExit("INT8 target gate failed")

    # Copy compact evidence/model outputs into Kaggle's downloadable working directory.
    for name in ["report.json", "generation_report.json", "checkpoint_latest.pt", "open_growth_rsnn_code_v1_int8.npz"]:
        src = SRC / "code_model/output" / name
        if src.exists():
            target = EXPORT / ("open_growth_rsnn_code_v2_int8.npz" if name == "open_growth_rsnn_code_v1_int8.npz" else name)
            shutil.copy2(src, target)
    shutil.copy2(SRC / "code_model/data/provenance.json", EXPORT / "provenance.json")

    status = {
        "status": "TRAINING_COMPLETE",
        "source_commit": SOURCE_COMMIT,
        "branch": BRANCH,
        "gpu": torch.cuda.get_device_name(0),
        "parameters": int(report["total_parameters"]),
        "train_examples": int(provenance["train"]["examples"]),
        "valid_examples": int(provenance["valid"]["examples"]),
        "initial_val_loss": float(report["initial_val_loss"]),
        "final_val_loss": float(report["final_val_loss"]),
        "loss_improvement": float(report["loss_improvement"]),
        "active_sparse_fraction": float(report["active_sparse_fraction"]),
        "syntax_rate": float(gen["syntax_rate"]),
        "python_keyword_rate": float(gen["python_keyword_rate"]),
        "elapsed_seconds_total": time.time() - started,
    }
    (EXPORT / "kaggle_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("CODE_V2_COMPLETE", json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
