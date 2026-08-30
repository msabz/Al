#!/usr/bin/env python3
"""Reviewed MAI5 v3 DeepMind-only Kaggle release runner.

This wrapper deliberately reuses the already-reviewed DeepMind-only training
patches, but replaces the obsolete audit patching stage with the current
per-family official-DeepMind audit contract.
"""

import json
import pathlib
import re
import urllib.request

SOURCE_COMMIT = "__SOURCE_COMMIT__"
if not re.fullmatch(r"[0-9a-fA-F]{40}", SOURCE_COMMIT):
    raise RuntimeError(f"SOURCE_COMMIT injection invalid: {SOURCE_COMMIT!r}")

WORK_DIR = pathlib.Path("/kaggle/working")
WORK_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = f"https://raw.githubusercontent.com/msabz/Al/{SOURCE_COMMIT}/kaggle/deepmind_only_train.py"
BASE_PATH = WORK_DIR / "_deepmind_only_v2_library.py"
urllib.request.urlretrieve(BASE_URL, BASE_PATH)
base_text = BASE_PATH.read_text().replace("__SOURCE_COMMIT__", SOURCE_COMMIT)

main_block = "\n\ntry:\n    core.main()\nfinally:\n    cleanup_large_temporaries()"
if main_block not in base_text:
    raise RuntimeError("Could not isolate reviewed DeepMind-only library from its old main block")
base_text = base_text.replace(main_block, "\n", 1)

ns = {"__name__": "mathai_deepmind_v2_library", "__file__": str(BASE_PATH)}
exec(compile(base_text, str(BASE_PATH), "exec"), ns)

core = ns["core"]
cleanup_large_temporaries = ns["cleanup_large_temporaries"]
original_prepare_source = ns["_original_prepare_source"]

EXPECTED_AUDIT_SCHEMA = "DEEPMIND_ONLY_PER_FAMILY_V2"
EXPECTED_BANKS = {
    "interpolate_linear_1d",
    "interpolate_linear_2d",
    "interpolate_polynomial",
    "extrapolate_polynomial",
}


def prepare_source_v3(log_fh):
    actual = original_prepare_source(log_fh)
    audit_path = core.ROOT / "colab/generalization_audit.py"
    text = audit_path.read_text()
    required = (
        'AUDIT_SCHEMA = "DEEPMIND_ONLY_PER_FAMILY_V2"',
        "def build_official_bank(",
        "RANDOM_BASELINE_SEEDS",
        'parser.add_argument("--validate-banks-only"',
        '"interpolate_linear_2d"',
        '"extrapolate_polynomial"',
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"v3 audit contract missing markers: {missing}")
    obsolete = ("def build_strict_ood(", "def build_fresh_iid(")
    present = [marker for marker in obsolete if marker in text]
    if present:
        raise RuntimeError(f"Obsolete duplicated audit banks still present: {present}")
    print("[V3] Audit contract verified: four independent per-family official DeepMind banks", flush=True)
    print("[V3] Random baseline contract: median of five fresh MAI5 initializations", flush=True)
    return actual


core.prepare_source = prepare_source_v3


def write_v3_evidence(audit):
    lines = [
        "Math AI v5 — MAI5 v3 Learning Evidence",
        "========================================",
        f"GENERALIZATION VERDICT: {audit.get('verdict', 'UNKNOWN')}",
        f"AUDIT SCHEMA: {audit.get('schema', 'UNKNOWN')}",
        "",
        "Training source: official pre-generated DeepMind Mathematics Dataset only.",
        "Audit source: official DeepMind test/test_extra generators only.",
        "Project synthetic training/audit examples: 0.",
        "Each bank is compared against the median of five fresh random MAI5 networks.",
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


core.write_evidence = write_v3_evidence


def validate_release_outputs():
    audit = json.loads(core.AUDIT.read_text())
    if audit.get("schema") != EXPECTED_AUDIT_SCHEMA:
        raise RuntimeError(f"Unexpected audit schema: {audit.get('schema')}")
    banks = audit.get("banks", {})
    if set(banks) != EXPECTED_BANKS:
        raise RuntimeError(f"Unexpected audit bank set: {sorted(banks)}")

    report = json.loads(core.REPORT.read_text())
    report.setdefault("model", {})["format"] = "MAI5 v3"
    report.setdefault("training", {})["data_contract"] = (
        "100% official pre-generated DeepMind train-easy/train-medium/train-hard; project synthetic=0"
    )
    report.setdefault("verification", {})["generalization_schema"] = EXPECTED_AUDIT_SCHEMA
    report["verification"]["generalization_banks"] = sorted(EXPECTED_BANKS)
    core.REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("[V3] Release metadata verified and normalized to MAI5 v3", flush=True)


try:
    core.main()
    validate_release_outputs()
finally:
    cleanup_large_temporaries()
