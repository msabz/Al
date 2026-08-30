#!/usr/bin/env python3
"""Autonomous Kaggle GPU runner for Math AI v5.

Kaggle owns the expensive GPU work. GitHub Actions later downloads the resulting
MAI5 + evidence and performs the Android build. No Android/Gradle work is done
inside the Kaggle GPU session.
"""

import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

SOURCE_REPO = "https://github.com/msabz/Al.git"
SOURCE_BRANCH = "feat/v5-deepmind-colab"
SOURCE_COMMIT = "__SOURCE_COMMIT__"  # replaced by GitHub Actions before kernel push

TOTAL_STEPS = 30_000
BATCH_SIZE = 0  # 0 = benchmark GPU and choose fastest batch
LEARNING_RATE = 2.0e-4
MIN_LEARNING_RATE = 2.0e-5
WARMUP_STEPS = 800
CONSISTENCY_WEIGHT = 0.03
DEEPMIND_RATIO = 0.65
CHECKPOINT_EVERY = 500
DEEPMIND_PER_FILE = 60_000
SYNTHETIC_POOL_SIZE = 260_000

WORK = pathlib.Path("/kaggle/working")
ROOT = WORK / "Al"
WORKING_MODEL = WORK / "math_ai_v5_working.mai5"
BEST_MODEL = WORK / "math_ai_v5_best.mai5"
AUDIT = WORK / "generalization_audit.json"
REPORT = WORK / "training_report.json"
EVIDENCE = WORK / "LEARNING_EVIDENCE.txt"
INTEROP = WORK / "v5_interop_expected.tsv"
CONSOLE = WORK / "training_console.log"
STATUS = WORK / "kaggle_job_status.json"


def banner(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88, flush=True)


def replace_setting(text, name, value_repr):
    pattern = rf"(?m)^{re.escape(name)}\s*=\s*.*$"
    updated, count = re.subn(pattern, f"{name} = {value_repr}", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not patch setting {name}")
    return updated


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        return out
    except Exception as exc:
        return f"unavailable: {exc}"


def run_stream(cmd, log_fh, cwd=None):
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_fh.write(line)
        log_fh.flush()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(map(str, cmd))}")


def prepare_source(log_fh):
    banner("[KAGGLE 1/7] تثبيت نسخة المصدر الدقيقة من GitHub")
    if ROOT.exists():
        shutil.rmtree(ROOT)
    run_stream(
        ["git", "clone", "-q", "--branch", SOURCE_BRANCH, "--single-branch", SOURCE_REPO, str(ROOT)],
        log_fh,
    )
    if SOURCE_COMMIT and SOURCE_COMMIT != "__SOURCE_COMMIT__":
        run_stream(["git", "-C", str(ROOT), "checkout", "-q", SOURCE_COMMIT], log_fh)
    actual = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    print("Source commit:", actual)
    if SOURCE_COMMIT not in ("", "__SOURCE_COMMIT__") and actual != SOURCE_COMMIT:
        raise RuntimeError(f"Source commit mismatch: expected {SOURCE_COMMIT}, got {actual}")

    # Make the shared Colab/Python runtime portable to Kaggle without changing the
    # Android model contract. Only filesystem paths and interactive downloads differ.
    base = ROOT / "colab/train_v5_deepmind.py"
    text = base.read_text().replace("/content", "/kaggle/working")
    text = replace_setting(text, "RESUME_FROM_MAI5", "False")
    text = replace_setting(text, "AUTO_DOWNLOAD_AT_END", "False")
    base.write_text(text)
    return actual


def prepare_turbo_worker():
    banner("[KAGGLE 2/7] تجهيز Turbo GPU worker")
    src_path = ROOT / "colab/turbo_train_v5.py"
    src = src_path.read_text().replace("/content", "/kaggle/working")
    src = src.replace("Google Colab", "Kaggle GPU")
    src = src.replace("In Colab choose Runtime > Change runtime type > T4 GPU.", "Kaggle kernel must use a CUDA GPU.")
    for name, value in {
        "TOTAL_STEPS": str(TOTAL_STEPS),
        "BATCH_SIZE": str(BATCH_SIZE),
        "LEARNING_RATE": repr(LEARNING_RATE),
        "MIN_LEARNING_RATE": repr(MIN_LEARNING_RATE),
        "WARMUP_STEPS": str(WARMUP_STEPS),
        "CONSISTENCY_WEIGHT": repr(CONSISTENCY_WEIGHT),
        "DEEPMIND_RATIO": repr(DEEPMIND_RATIO),
        "CHECKPOINT_EVERY": str(CHECKPOINT_EVERY),
        "DEEPMIND_PER_FILE": str(DEEPMIND_PER_FILE),
        "SYNTHETIC_POOL_SIZE": str(SYNTHETIC_POOL_SIZE),
        "RESUME_FROM_MAI5": "False",
        "AUTO_DOWNLOAD_AT_END": "False",
        "OUTPUT_FILE": repr(str(WORKING_MODEL)),
    }.items():
        src = replace_setting(src, name, value)
    worker = ROOT / "colab/_kaggle_turbo_worker.py"
    worker.write_text(src)
    print("Training configuration:")
    print(f"  steps={TOTAL_STEPS:,} batch=auto peak_lr={LEARNING_RATE:g}")
    print(f"  DeepMind={DEEPMIND_RATIO*100:.0f}% synthetic={(1-DEEPMIND_RATIO)*100:.0f}%")
    print(f"  DeepMind pre-encode cap={DEEPMIND_PER_FILE:,}/file synthetic_pool={SYNTHETIC_POOL_SIZE:,}")
    return worker


def train_and_select(worker, log_fh):
    banner("[KAGGLE 3/7] تدريب GPU واختيار أفضل Holdout checkpoint")
    best_rmse = float("inf")
    best_metrics = None
    final_metrics = None
    checkpoint_re = re.compile(
        r"HOLDOUT rmse=([0-9.eE+-]+) mae=([0-9.eE+-]+) ±1=([0-9.eE+-]+)% state=([0-9.eE+-]+)%"
    )
    final_re = re.compile(
        r"Holdout RMSE=([0-9.eE+-]+)\s+MAE=([0-9.eE+-]+)\s+within ±1=([0-9.eE+-]+)%\s+state accuracy=([0-9.eE+-]+)%"
    )
    proc = subprocess.Popen(
        [sys.executable, "-u", str(worker)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_fh.write(line)
        log_fh.flush()
        match = checkpoint_re.search(line) or final_re.search(line)
        if match:
            metrics = {
                "rmse": float(match.group(1)),
                "mae": float(match.group(2)),
                "within_one_percent": float(match.group(3)),
                "state_accuracy_percent": float(match.group(4)),
            }
            final_metrics = metrics
            if WORKING_MODEL.is_file() and metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_metrics = dict(metrics)
                shutil.copy2(WORKING_MODEL, BEST_MODEL)
                print(f"KAGGLE_BEST_CHECKPOINT RMSE={best_rmse:.6f}", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Turbo trainer failed with exit code {rc}")
    if not BEST_MODEL.is_file():
        if not WORKING_MODEL.is_file():
            raise RuntimeError("Training completed without a MAI5 checkpoint")
        shutil.copy2(WORKING_MODEL, BEST_MODEL)
        best_metrics = final_metrics
    print("Best MAI5:", BEST_MODEL, BEST_MODEL.stat().st_size, "bytes")
    print("Best Holdout:", best_metrics)
    return best_metrics, final_metrics


def run_audit(log_fh):
    banner("[KAGGLE 4/7] Generalization / anti-memorization audit")
    audit_script = ROOT / "colab/generalization_audit.py"
    run_stream(
        [
            sys.executable,
            "-u",
            str(audit_script),
            "--root",
            str(ROOT),
            "--model",
            str(BEST_MODEL),
            "--output",
            str(AUDIT),
        ],
        log_fh,
        cwd=ROOT,
    )
    audit = json.loads(AUDIT.read_text())
    print("GENERALIZATION VERDICT:", audit.get("verdict"))
    return audit


def generate_interop(log_fh):
    banner("[KAGGLE 5/7] توليد Python reference لاختبار Kotlin byte-for-byte")
    trainer = ROOT / "colab/train_v5_deepmind.py"
    src = trainer.read_text()
    src = replace_setting(src, "RESUME_FROM_MAI5", "False")
    src = replace_setting(src, "AUTO_DOWNLOAD_AT_END", "False")
    prefix = src.split("# ========================= TRAIN =========================", 1)[0]
    ns = {"__name__": "kaggle_mai5_interop", "__file__": str(trainer)}
    exec(compile(prefix, str(trainer), "exec"), ns)
    ns["load_mai5"](str(BEST_MODEL))
    torch = ns["torch"]
    np = ns["np"]
    model = ns["model"]
    device = ns["device"]
    encode = ns["encode"]
    root_scale = ns["ROOT_SCALE"]
    equations = [
        "2x+4=10",
        "(x-2)*(x+3)=0",
        "ln(2x+1)=1.60943791",
        "2x+3y=5;x-y=1",
        "0*x=1",
        "0*x=0",
    ]
    rows = ["# equation\tfamily\tstate\tslots\tpresence\tstate_probabilities"]
    model.eval()
    with torch.no_grad():
        for equation in equations:
            k, n, d, fam, normalized = encode(equation)
            out = model(
                torch.tensor(k[None, :], device=device, dtype=torch.long),
                torch.tensor(n[None, :], device=device, dtype=torch.float32),
                torch.tensor(d[None, :], device=device, dtype=torch.float32),
                torch.tensor([fam], device=device, dtype=torch.long),
            )[0]
            slots = (out[:5] * root_scale).detach().cpu().numpy().astype(float)
            presence = torch.sigmoid(out[5:10]).detach().cpu().numpy().astype(float)
            state_probs = torch.softmax(out[10:14], dim=0).detach().cpu().numpy().astype(float)
            state = int(np.argmax(state_probs))
            csv = lambda arr: ",".join(f"{float(x):.9g}" for x in arr)
            rows.append("\t".join([normalized, str(int(fam)), str(state), csv(slots), csv(presence), csv(state_probs)]))
    INTEROP.write_text("\n".join(rows) + "\n")
    print("Interop reference:", INTEROP)
    log_fh.write(f"Interop reference: {INTEROP}\n")
    return equations


def write_evidence(audit):
    trained = audit.get("trained", {})
    gains = audit.get("improvement_vs_random", {})
    consistency = audit.get("equivalence_consistency", {})
    lines = [
        "Math AI v5 — Kaggle Learning Evidence",
        "=======================================",
        f"GENERALIZATION VERDICT: {audit.get('verdict', 'UNKNOWN')}",
        "",
        "This is empirical evidence, not a mathematical proof that memorization is impossible.",
        "Held-out distributions/forms below were not used by the optimizer.",
        "",
    ]
    for name in ("fresh_iid", "strict_ood", "deepmind_interpolate", "deepmind_extrapolate"):
        metric = trained.get(name, {})
        if metric:
            lines.append(
                f"{name}: RMSE={metric.get('rmse', float('nan')):.4f}, "
                f"±1={metric.get('within_one_ratio', 0.0)*100:.2f}%, "
                f"state={metric.get('state_accuracy', 0.0)*100:.2f}%, "
                f"vs-random improvement={gains.get(name, 0.0)*100:.2f}%"
            )
    lines.extend(
        [
            "",
            f"Equivalent-form mean slot delta: {consistency.get('mean_slot_delta', float('nan')):.4f}",
            f"Warnings: {audit.get('warnings', [])}",
        ]
    )
    EVIDENCE.write_text("\n".join(lines) + "\n")


def main():
    started = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    for path in (WORKING_MODEL, BEST_MODEL, AUDIT, REPORT, EVIDENCE, INTEROP, STATUS):
        if path.exists():
            path.unlink()
    with CONSOLE.open("w", encoding="utf-8") as log_fh:
        banner("Math AI v5 — AUTONOMOUS KAGGLE GPU TRAINING")
        print("GPU:", gpu_info())
        print("Kaggle handles only compute-heavy training/audit; GitHub will build Android afterwards.")
        source_commit = prepare_source(log_fh)
        worker = prepare_turbo_worker()
        best_metrics, final_metrics = train_and_select(worker, log_fh)
        audit = run_audit(log_fh)
        equations = generate_interop(log_fh)
        write_evidence(audit)

        banner("[KAGGLE 6/7] كتابة تقرير قابل للتدقيق")
        report = {
            "pipeline": "Math AI v5 autonomous Kaggle GPU training",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_repository": SOURCE_REPO,
            "source_branch": SOURCE_BRANCH,
            "source_commit": source_commit,
            "gpu": gpu_info(),
            "training": {
                "total_steps": TOTAL_STEPS,
                "batch_size": "auto-throughput-benchmark" if BATCH_SIZE == 0 else BATCH_SIZE,
                "peak_learning_rate": LEARNING_RATE,
                "min_learning_rate": MIN_LEARNING_RATE,
                "warmup_steps": WARMUP_STEPS,
                "consistency_weight": CONSISTENCY_WEIGHT,
                "deepmind_ratio": DEEPMIND_RATIO,
                "deepmind_modules": ["algebra__linear_1d", "algebra__linear_2d", "algebra__polynomial_roots"],
                "deepmind_per_file_cap": DEEPMIND_PER_FILE,
                "synthetic_pool_size": SYNTHETIC_POOL_SIZE,
                "best_holdout": best_metrics,
                "final_holdout": final_metrics,
                "elapsed_seconds": round(time.time() - started, 1),
            },
            "model": {
                "file": BEST_MODEL.name,
                "format": "MAI5 v1",
                "bytes": BEST_MODEL.stat().st_size,
                "sha256": sha256(BEST_MODEL),
            },
            "verification": {
                "generalization_verdict": audit.get("verdict"),
                "generalization_warnings": audit.get("warnings", []),
                "interop_reference_equations": equations,
                "android_build": "pending GitHub quality gate",
            },
        }
        REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        STATUS.write_text(
            json.dumps(
                {
                    "status": "TRAINING_COMPLETE",
                    "verdict": audit.get("verdict", "UNKNOWN"),
                    "source_commit": source_commit,
                    "model_sha256": report["model"]["sha256"],
                },
                indent=2,
            )
        )

        banner("[KAGGLE 7/7] GPU job complete — handoff to GitHub")
        print("Verdict      :", audit.get("verdict"))
        print("Best MAI5    :", BEST_MODEL)
        print("Audit        :", AUDIT)
        print("Interop      :", INTEROP)
        print("Report       :", REPORT)
        print("GitHub Actions will now download these files and run the Kotlin/Android gate.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        STATUS.write_text(
            json.dumps(
                {
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
        )
        print("KAGGLE_JOB_FAILED:", repr(exc), flush=True)
        raise
