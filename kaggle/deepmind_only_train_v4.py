#!/usr/bin/env python3
"""Reviewed MAI5 v4 DeepMind-only Kaggle release runner.

Model change vs v3 is intentionally singular: the polynomial head reuses logits
5..9 as a 5-way root-cardinality classifier and decodes exactly k root slots by
smallest canonical polynomial residual. Other families retain presence semantics.
Training/audit data remain official DeepMind only.
"""

import json
import os
import pathlib
import re
import urllib.request

SOURCE_COMMIT = globals().get("SOURCE_COMMIT")
if not isinstance(SOURCE_COMMIT, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", SOURCE_COMMIT):
    raise RuntimeError(f"SOURCE_COMMIT runtime handoff invalid: {SOURCE_COMMIT!r}")

KAGGLE_WORK = pathlib.Path("/kaggle/working")
if KAGGLE_WORK.is_dir():
    WORK_DIR = KAGGLE_WORK
else:
    WORK_DIR = pathlib.Path(os.environ.get("MATHAI_WORK_DIR", "/tmp/mathai-kaggle-v4-preflight"))
    WORK_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = f"https://raw.githubusercontent.com/msabz/Al/{SOURCE_COMMIT}/kaggle/deepmind_only_train.py"
BASE_PATH = WORK_DIR / "_deepmind_only_v2_library.py"
urllib.request.urlretrieve(BASE_URL, BASE_PATH)
base_text = BASE_PATH.read_text().replace("__SOURCE_COMMIT__", SOURCE_COMMIT)
base_text = base_text.replace("/kaggle/working", WORK_DIR.as_posix())

main_block = "\n\ntry:\n    core.main()\nfinally:\n    cleanup_large_temporaries()"
if main_block not in base_text:
    raise RuntimeError("Could not isolate reviewed DeepMind-only library from its old main block")
base_text = base_text.replace(main_block, "\n", 1)

ns = {"__name__": "mathai_deepmind_v2_library", "__file__": str(BASE_PATH)}
exec(compile(base_text, str(BASE_PATH), "exec"), ns)

core = ns["core"]
core.WORK = WORK_DIR
core.ROOT = WORK_DIR / "Al"
core.WORKING_MODEL = WORK_DIR / "math_ai_v5_working.mai5"
core.BEST_MODEL = WORK_DIR / "math_ai_v5_best.mai5"
core.AUDIT = WORK_DIR / "generalization_audit.json"
core.REPORT = WORK_DIR / "training_report.json"
core.EVIDENCE = WORK_DIR / "LEARNING_EVIDENCE.txt"
core.INTEROP = WORK_DIR / "v5_interop_expected.tsv"
core.CONSOLE = WORK_DIR / "training_console.log"
core.STATUS = WORK_DIR / "kaggle_job_status.json"

cleanup_large_temporaries = ns["cleanup_large_temporaries"]
original_prepare_source = ns["_original_prepare_source"]

EXPECTED_AUDIT_SCHEMA = "DEEPMIND_ONLY_PER_FAMILY_V2"
EXPECTED_BANKS = {
    "interpolate_linear_1d",
    "interpolate_linear_2d",
    "interpolate_polynomial",
    "extrapolate_polynomial",
}


def enforce_deterministic_audit(audit_path):
    """Make official DeepMind audit banks reproducible and runtime paths portable."""
    text = audit_path.read_text()

    marker = "DEEPMIND_BANK_GLOBAL_RESEED"
    if marker not in text:
        anchor = '''    try:\n        ns["synthetic"] = project_synthetic_forbidden\n        if spec["split"] == "interpolate":\n'''
        replacement = '''    try:\n        ns["synthetic"] = project_synthetic_forbidden\n        # DEEPMIND_BANK_GLOBAL_RESEED: official generator modules also consume\n        # process-global RNG state; fully reseed per bank for reproducibility.\n        bank_seed = int(spec["seed"])\n        random.seed(bank_seed)\n        ns["np"].random.seed(bank_seed & 0xFFFFFFFF)\n        ns["torch"].manual_seed(bank_seed)\n        if spec["split"] == "interpolate":\n'''
        if text.count(anchor) != 1:
            raise RuntimeError("v4 deterministic-audit insertion anchor changed")
        text = text.replace(anchor, replacement, 1)
        old_rng = '        rng = random.Random(int(spec["seed"]))'
        if text.count(old_rng) != 1:
            raise RuntimeError("v4 deterministic-audit RNG anchor changed")
        text = text.replace(old_rng, '        rng = random.Random(bank_seed)', 1)

    portable_marker = "DEEPMIND_AUDIT_RUNTIME_PORTABLE"
    if portable_marker not in text:
        old_portable = '    src = src.replace("/content", runtime_dir.as_posix())'
        new_portable = '''    # DEEPMIND_AUDIT_RUNTIME_PORTABLE: the reviewed Kaggle adapter may rewrite\n    # trainer /content paths to /kaggle/working; redirect either spelling into\n    # this audit's writable runtime directory before executing trainer imports.\n    src = src.replace("/content", runtime_dir.as_posix()).replace("/kaggle/working", runtime_dir.as_posix())'''
        if text.count(old_portable) != 1:
            raise RuntimeError("v4 audit runtime portability anchor changed")
        text = text.replace(old_portable, new_portable, 1)

    audit_path.write_text(text)


def prepare_source_v4(log_fh):
    actual = original_prepare_source(log_fh)

    trainer = core.ROOT / "colab/train_v5_deepmind.py"
    trainer_text = trainer.read_text()
    required_trainer = (
        "MAGIC=0x4D414935; VERSION=4",
        "def polynomial_active_indices(",
        "count_target=root_count[poly_ids].clamp(1,ROOT_SLOTS)-1",
        "nonpoly_match=(families[ids]!=POLYNOMIAL)",
    )
    missing = [marker for marker in required_trainer if marker not in trainer_text]
    if missing:
        raise RuntimeError(f"v4 trainer contract missing markers: {missing}")

    audit_path = core.ROOT / "colab/generalization_audit.py"
    enforce_deterministic_audit(audit_path)
    audit_text = audit_path.read_text()
    required_audit = (
        'AUDIT_SCHEMA = "DEEPMIND_ONLY_PER_FAMILY_V2"',
        "def build_official_bank(",
        "RANDOM_BASELINE_SEEDS",
        'parser.add_argument("--validate-banks-only"',
        '"interpolate_linear_2d"',
        '"extrapolate_polynomial"',
        'root_count_probs',
        'DEEPMIND_BANK_GLOBAL_RESEED',
        'DEEPMIND_AUDIT_RUNTIME_PORTABLE',
    )
    missing = [marker for marker in required_audit if marker not in audit_text]
    if missing:
        raise RuntimeError(f"v4 audit contract missing markers: {missing}")

    print("[V4] Polynomial contract: 5-way cardinality CE + residual-ranked top-k", flush=True)
    print("[V4] Audit contract: deterministic official DeepMind per-family banks", flush=True)
    print("[V4] Audit runtime: portable across Kaggle/GitHub paths", flush=True)
    print("[V4] Project synthetic training/audit examples: 0", flush=True)
    return actual


core.prepare_source = prepare_source_v4


def generate_interop_v4(log_fh):
    """Generate the exact Python sidecar semantics consumed by Kotlin v4 tests."""
    trainer = core.ROOT / "colab/train_v5_deepmind.py"
    src = trainer.read_text()
    src = core.replace_setting(src, "RESUME_FROM_MAI5", "False")
    src = core.replace_setting(src, "AUTO_DOWNLOAD_AT_END", "False")
    prefix = src.split("# ========================= TRAIN =========================", 1)[0]
    runtime = {"__name__": "kaggle_mai5_v4_interop", "__file__": str(trainer)}
    exec(compile(prefix, str(trainer), "exec"), runtime)
    runtime["load_mai5"](str(core.BEST_MODEL))

    torch = runtime["torch"]
    np = runtime["np"]
    model = runtime["model"]
    device = runtime["device"]
    encode = runtime["encode"]
    root_scale = runtime["ROOT_SCALE"]
    polynomial = runtime["POLYNOMIAL"]
    active_indices = runtime["polynomial_active_indices"]

    equations = [
        "2x+4=10",
        "(x-2)*(x+3)=0",
        "x^3-6*x^2+11*x-6=0",
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
            kinds = torch.tensor(k[None, :], device=device, dtype=torch.long)
            numeric = torch.tensor(n[None, :], device=device, dtype=torch.float32)
            depth = torch.tensor(d[None, :], device=device, dtype=torch.float32)
            out = model(kinds, numeric, depth, torch.tensor([fam], device=device, dtype=torch.long))[0]
            slots = (out[:5] * root_scale).detach().cpu().numpy().astype(float)
            state_probs = torch.softmax(out[10:14], dim=0).detach().cpu().numpy().astype(float)
            state = int(np.argmax(state_probs))
            if fam == polynomial:
                presence = np.zeros(5, dtype=float)
                if state == runtime["FINITE"]:
                    chosen = active_indices(out, numeric[0]).detach().cpu().numpy().astype(int).tolist()
                    presence[chosen] = 1.0
            else:
                presence = torch.sigmoid(out[5:10]).detach().cpu().numpy().astype(float)
            csv = lambda arr: ",".join(f"{float(x):.9g}" for x in arr)
            rows.append("\t".join([normalized, str(int(fam)), str(state), csv(slots), csv(presence), csv(state_probs)]))

    core.INTEROP.write_text("\n".join(rows) + "\n")
    print("[V4] Interop reference:", core.INTEROP, flush=True)
    log_fh.write(f"V4 interop reference: {core.INTEROP}\n")
    return equations


core.generate_interop = generate_interop_v4


def write_v4_evidence(audit):
    lines = [
        "Math AI v5 — MAI5 v4 Learning Evidence",
        "========================================",
        f"GENERALIZATION VERDICT: {audit.get('verdict', 'UNKNOWN')}",
        f"AUDIT SCHEMA: {audit.get('schema', 'UNKNOWN')}",
        "",
        "Polynomial output: learned 5-way root cardinality + residual-ranked top-k slots.",
        "Training source: official pre-generated DeepMind Mathematics Dataset only.",
        "Audit source: official DeepMind test/test_extra generators only.",
        "Project synthetic training/audit examples: 0.",
        "Each family is compared against the median of five fresh random MAI5 networks.",
        "",
    ]
    banks = audit.get("banks", {})
    for name in sorted(banks):
        row = banks[name]
        trained = row.get("trained", {})
        gains = row.get("gains", {})
        lines.append(
            f"{name}: passed={row.get('passed')} "
            f"RMSE={trained.get('rmse', float('nan')):.6g} "
            f"MAE={trained.get('mae', float('nan')):.6g} "
            f"±1={trained.get('within_one_ratio', 0.0)*100:.2f}% "
            f"count={trained.get('root_count_accuracy', 0.0)*100:.2f}% "
            f"state={trained.get('state_accuracy', 0.0)*100:.2f}% "
            f"RMSE_gain={gains.get('rmse_gain', 0.0)*100:.2f}% "
            f"MAE_gain={gains.get('mae_gain', 0.0)*100:.2f}%"
        )
    lines.extend(["", f"Warnings: {audit.get('warnings', [])}"])
    core.EVIDENCE.write_text("\n".join(lines) + "\n")


core.write_evidence = write_v4_evidence


def validate_release_outputs():
    audit = json.loads(core.AUDIT.read_text())
    if audit.get("schema") != EXPECTED_AUDIT_SCHEMA:
        raise RuntimeError(f"Unexpected audit schema: {audit.get('schema')}")
    banks = audit.get("banks", {})
    if set(banks) != EXPECTED_BANKS:
        raise RuntimeError(f"Unexpected audit bank set: {sorted(banks)}")

    report = json.loads(core.REPORT.read_text())
    report.setdefault("model", {})["format"] = "MAI5 v4"
    report["model"]["polynomial_cardinality"] = "5-way root-count classifier; residual-ranked top-k"
    report.setdefault("training", {})["data_contract"] = (
        "100% official pre-generated DeepMind train-easy/train-medium/train-hard; project synthetic=0"
    )
    report.setdefault("verification", {})["generalization_schema"] = EXPECTED_AUDIT_SCHEMA
    report["verification"]["generalization_banks"] = sorted(EXPECTED_BANKS)
    core.REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("[V4] Release metadata verified and normalized to MAI5 v4", flush=True)


try:
    core.main()
    validate_release_outputs()
finally:
    cleanup_large_temporaries()
