# Math AI v5 — verified one-click launcher
# This wrapper runs the normal factory without its early download, then performs
# an independent generalization audit and downloads ONE final bundle.

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
BASE_FACTORY = HERE / "one_click_factory.py"

# Optional overrides from the Colab cell/environment.
OVERRIDES = {
    "TOTAL_STEPS": os.environ.get("MATHAI_STEPS"),
    "BATCH_SIZE": os.environ.get("MATHAI_BATCH"),
    "LEARNING_RATE": os.environ.get("MATHAI_LR"),
    "DEEPMIND_RATIO": os.environ.get("MATHAI_DEEPMIND_RATIO"),
    "CHECKPOINT_EVERY": os.environ.get("MATHAI_CHECKPOINT_EVERY"),
}


def replace_setting(text, name, raw_value):
    pattern = rf"(?m)^{re.escape(name)}\s*=\s*.*$"
    updated, count = re.subn(pattern, f"{name} = {raw_value}", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not override factory setting: {name}")
    return updated


source = BASE_FACTORY.read_text()
for name, value in OVERRIDES.items():
    if value is None or value == "":
        continue
    if name in ("TOTAL_STEPS", "BATCH_SIZE", "CHECKPOINT_EVERY"):
        value = str(int(value))
    else:
        value = repr(float(value))
    source = replace_setting(source, name, value)

# Prevent the base factory from downloading an un-audited bundle.
download_call = "files.download(str(BUNDLE))"
if download_call not in source:
    raise RuntimeError("Base factory download hook changed; refusing to bypass audit")
source = source.replace(download_call, 'print("Base factory complete; running generalization audit before download...")', 1)

factory_ns = {"__name__": "math_ai_v5_verified_factory_base"}
exec(compile(source, str(BASE_FACTORY), "exec"), factory_ns)

root = pathlib.Path(factory_ns["ROOT"])
best_model = pathlib.Path(factory_ns["BEST_MODEL"])
report_path = pathlib.Path(factory_ns["REPORT"])
out_dir = pathlib.Path("/content/MathAI-v5-output")
audit_path = pathlib.Path("/content/generalization_audit.json")

print("\n7) Running anti-memorization / generalization audit...")
subprocess.run(
    [
        sys.executable,
        str(root / "colab/generalization_audit.py"),
        "--root", str(root),
        "--model", str(best_model),
        "--output", str(audit_path),
    ],
    check=True,
)

audit = json.loads(audit_path.read_text())
verdict = audit.get("verdict", "UNKNOWN")

# Merge audit into the normal training report.
report = json.loads(report_path.read_text())
report.setdefault("verification", {})["generalization_audit"] = {
    "verdict": verdict,
    "improvement_vs_random": audit.get("improvement_vs_random"),
    "equivalence_consistency": audit.get("equivalence_consistency"),
    "warnings": audit.get("warnings", []),
    "important_limit": audit.get("important_limit"),
}
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

out_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(audit_path, out_dir / "generalization_audit.json")
shutil.copy2(report_path, out_dir / "training_report.json")

trained = audit.get("trained", {})
gains = audit.get("improvement_vs_random", {})
consistency = audit.get("equivalence_consistency", {})
summary_lines = [
    "Math AI v5 — Learning Evidence",
    "================================",
    f"GENERALIZATION VERDICT: {verdict}",
    "",
    "This is not a mathematical proof that memorization is impossible.",
    "It is an empirical audit on data/forms not used by the optimizer.",
    "",
]
for name in ("fresh_iid", "strict_ood", "deepmind_interpolate", "deepmind_extrapolate"):
    metric = trained.get(name, {})
    gain = gains.get(name)
    if metric:
        summary_lines.append(
            f"{name}: RMSE={metric.get('rmse', float('nan')):.4f}, "
            f"±1={metric.get('within_one_ratio', 0.0)*100:.2f}%, "
            f"state={metric.get('state_accuracy', 0.0)*100:.2f}%, "
            f"vs-random improvement={(gain or 0.0)*100:.2f}%"
        )
summary_lines += [
    "",
    f"Equivalent-form mean slot delta: {consistency.get('mean_slot_delta', float('nan')):.4f}",
    f"Warnings: {audit.get('warnings', [])}",
]
(out_dir / "LEARNING_EVIDENCE.txt").write_text("\n".join(summary_lines) + "\n")

# Rebuild one final bundle AFTER the audit.
verified_zip_base = pathlib.Path("/content") / (
    "MathAI-v5-VERIFIED-output" if verdict == "PASS" else f"MathAI-v5-{verdict}-output"
)
verified_zip = verified_zip_base.with_suffix(".zip")
if verified_zip.exists():
    verified_zip.unlink()
shutil.make_archive(str(verified_zip_base), "zip", out_dir)

print("\n=== VERIFIED FACTORY COMPLETE ===")
print("Generalization verdict:", verdict)
if verdict == "PASS":
    print("PASS: model improved on fresh, OOD and official DeepMind held-out tests.")
elif verdict == "WARN":
    print("WARN: some held-out tests are good, but there is evidence of a generalization gap.")
else:
    print("FAIL: do NOT treat this checkpoint as a successfully generalized model yet.")
for warning in audit.get("warnings", []):
    print("  -", warning)
print("Final bundle:", verified_zip)

from google.colab import files
files.download(str(verified_zip))
