# Math AI v5 — TURBO + VERIFIED one-click launcher
# One Colab cell: maximize GPU use, explain every phase, verify generalization,
# embed the best MAI5 into Android, build APK, and download one final ZIP.

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
BASE_FACTORY = HERE / "one_click_factory.py"

print("""
==============================================================================
Math AI v5 — TURBO VERIFIED FACTORY
==============================================================================
ما سيحدث:
  1) فحص CPU/GPU/RAM/VRAM.
  2) تنزيل DeepMind Mathematics Dataset الرسمي الجاهز بدل توليده بـSymPy كل خطوة.
  3) استخراج وحدات المعادلات المناسبة فقط.
  4) تحويل مئات آلاف الأمثلة إلى RPN tensors مرة واحدة.
  5) تحميل pool مضغوط إلى VRAM.
  6) Benchmark تلقائي واختيار Batch يعطي أعلى throughput فعلي.
  7) تدريب AMP على GPU مع Warmup + Cosine LR + Gradient Clipping.
  8) Holdout دوري واختيار أفضل checkpoint، لا آخر checkpoint.
  9) اختبار تعميم مستقل ضد الحفظ.
 10) اختبار Python ↔ Kotlin لنفس MAI5.
 11) حقن أفضل نموذج داخل APK ثم بناء التطبيق.
 12) تنزيل ZIP: APK + MAI5 + training report + generalization evidence.
==============================================================================
""", flush=True)

source = BASE_FACTORY.read_text()


def replace_setting(text, name, raw_value):
    pattern = rf"(?m)^{re.escape(name)}\s*=\s*.*$"
    updated, count = re.subn(pattern, f"{name} = {raw_value}", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not override factory setting: {name}")
    return updated


defaults = {
    "TOTAL_STEPS": os.environ.get("MATHAI_STEPS", "30000"),
    "BATCH_SIZE": os.environ.get("MATHAI_BATCH", "0"),
    "LEARNING_RATE": os.environ.get("MATHAI_LR", "0.0002"),
    "CONSISTENCY_WEIGHT": os.environ.get("MATHAI_CONSISTENCY", "0.03"),
    "DEEPMIND_RATIO": os.environ.get("MATHAI_DEEPMIND_RATIO", "0.65"),
    "CHECKPOINT_EVERY": os.environ.get("MATHAI_CHECKPOINT_EVERY", "500"),
}
for name, raw in defaults.items():
    if name in ("TOTAL_STEPS", "BATCH_SIZE", "CHECKPOINT_EVERY"):
        raw = str(int(raw))
    else:
        raw = repr(float(raw))
    source = replace_setting(source, name, raw)

old_loader = 'trainer_src = (ROOT / "colab/train_v5_deepmind.py").read_text()'
new_loader = (
    'reference_trainer_src = (ROOT / "colab/train_v5_deepmind.py").read_text()\n'
    'trainer_src = (ROOT / "colab/turbo_train_v5.py").read_text()\n'
    'print("Training engine: TURBO GPU-resident DeepMind pipeline")'
)
if old_loader not in source:
    raise RuntimeError("Factory trainer loader changed")
source = source.replace(old_loader, new_loader, 1)

old_reference = 'reference_source = replace_setting(trainer_src, "RESUME_FROM_MAI5", "False")'
new_reference = 'reference_source = replace_setting(reference_trainer_src, "RESUME_FROM_MAI5", "False")'
if old_reference not in source:
    raise RuntimeError("Factory reference source hook changed")
source = source.replace(old_reference, new_reference, 1)

source = replace_setting(source, "RESUME_FROM_MAI5", "False")

download_call = 'files.download(str(BUNDLE))'
if download_call not in source:
    raise RuntimeError("Factory download hook changed")
source = source.replace(
    download_call,
    'print("Base build complete; now running independent generalization audit before download...")',
    1,
)

factory_ns = {"__name__": "math_ai_v5_turbo_factory_base"}
exec(compile(source, str(BASE_FACTORY), "exec"), factory_ns)

root = pathlib.Path(factory_ns["ROOT"])
best_model = pathlib.Path(factory_ns["BEST_MODEL"])
report_path = pathlib.Path(factory_ns["REPORT"])
out_dir = pathlib.Path("/content/MathAI-v5-output")
audit_path = pathlib.Path("/content/generalization_audit.json")

print("\n" + "=" * 78)
print("[9/10] اختبار التعميم ضد الحفظ")
print("=" * 78)
print("هذا الاختبار لا يستخدم بيانات optimizer.")
print("سيختبر: fresh IID + strict OOD + DeepMind interpolate + extrapolate + equivalent forms.")
print("إذا تحسن Train فقط وفشل هنا، النتيجة ستكون WARN/FAIL وليس PASS.", flush=True)

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

report = json.loads(report_path.read_text())
report.setdefault("training", {})["engine"] = "turbo_gpu_resident"
report["training"]["auto_batch_requested"] = int(defaults["BATCH_SIZE"]) == 0
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
    "This is empirical evidence, not a mathematical proof that memorization is impossible.",
    "The audit uses held-out distributions/forms that were not used by the optimizer.",
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

verified_zip_base = pathlib.Path("/content") / (
    "MathAI-v5-TURBO-VERIFIED-output" if verdict == "PASS"
    else f"MathAI-v5-TURBO-{verdict}-output"
)
verified_zip = verified_zip_base.with_suffix(".zip")
if verified_zip.exists():
    verified_zip.unlink()
shutil.make_archive(str(verified_zip_base), "zip", out_dir)

print("\n" + "=" * 78)
print("[10/10] النتيجة النهائية")
print("=" * 78)
print("GENERALIZATION VERDICT:", verdict)
if verdict == "PASS":
    print("PASS = يوجد دليل تجريبي قوي على تعلم قابل للتعميم خارج عينات التدريب.")
elif verdict == "WARN":
    print("WARN = هناك تعلم، لكن توجد فجوة تعميم يجب علاجها قبل اعتبار النموذج جاهزًا.")
else:
    print("FAIL = لا نعتبر النموذج ناجحًا حتى لو كان training loss منخفضًا.")
for warning in audit.get("warnings", []):
    print("  -", warning)
print("Final bundle:", verified_zip)
print("سيبدأ تنزيل ZIP الآن.", flush=True)

from google.colab import files
files.download(str(verified_zip))
