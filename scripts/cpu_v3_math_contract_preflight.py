#!/usr/bin/env python3
"""CPU-only proof that the v3 engineered representation is mathematically correct.

No project synthetic examples are created. Samples come only from the official
DeepMind Mathematics Dataset test generators. This does NOT train the model; it
validates the signal before any GPU runtime is spent.
"""
import math
import pathlib
import random
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINER = ROOT / "colab/train_v5_deepmind.py"
src = TRAINER.read_text()
# Avoid interactive/download behavior and execute definitions only.
src = re.sub(r"(?m)^RESUME_FROM_MAI5\s*=.*$", "RESUME_FROM_MAI5 = False", src, count=1)
src = re.sub(r"(?m)^AUTO_DOWNLOAD_AT_END\s*=.*$", "AUTO_DOWNLOAD_AT_END = False", src, count=1)
prefix = src.split("# ========================= TRAIN =========================", 1)[0]
ns = {"__name__": "v3_cpu_math_contract", "__file__": str(TRAINER)}
exec(compile(prefix, str(TRAINER), "exec"), ns)

assert ns["VERSION"] == 3
assert ns["POLYNOMIAL_FEATURE_SLOTS"] == 7
assert ns["SYSTEM_FEATURE_SLOTS"] == 9


def eval_poly(coeff, z):
    q = float(coeff[-1])
    for c in reversed(coeff[:-1]):
        q = q * z + float(c)
    return q


def official_examples(name, count, seed):
    old_modules = ns["dm_modules"]
    old_names = list(ns["DM_NAMES"])
    ns["dm_modules"] = ns["dm_algebra"].test()
    ns["DM_NAMES"] = [name]
    rng = random.Random(seed)
    out = []
    attempts = 0
    try:
        while len(out) < count and attempts < count * 500:
            attempts += 1
            try:
                e = ns["deepmind_example"](rng, allow_synthetic_fallback=False)
            except Exception:
                continue
            vals = list(e.get("roots", ())) + list(e.get("system", ()))
            if not vals or any((not math.isfinite(float(v))) or abs(float(v)) > 300.0 for v in vals):
                continue
            out.append(e)
    finally:
        ns["dm_modules"] = old_modules
        ns["DM_NAMES"] = old_names
    if len(out) != count:
        raise RuntimeError(f"official DeepMind {name}: only {len(out)}/{count} usable examples")
    return out


systems = official_examples("linear_2d", 96, 0xA1653001)
max_system_error = 0.0
max_invariant_error = 0.0
for e in systems:
    k, numeric, depth, fam, source = ns["encode"](e["eq"])
    if fam != ns["SYSTEM"]:
        raise AssertionError(("family", source, fam))
    if sum(int(x != 0) for x in k[:9]) != 9:
        raise AssertionError(("system feature count", source, k[:12].tolist()))
    a1,b1,c1,a2,b2,c2 = map(float, numeric[:6])
    det = a1*b2-a2*b1
    nx = c1*b2-c2*b1
    ny = a1*c2-a2*c1
    if abs(det) < 1e-8:
        raise AssertionError(("singular finite system", source, det))
    pred = [nx/det, ny/det]
    expected = [float(v)/ns["ROOT_SCALE"] for v in e["system"]]
    err = max(abs(pred[i]-expected[i]) for i in range(2))
    max_system_error = max(max_system_error, err)
    scale = max(abs(det),abs(nx),abs(ny))
    inv = [0.0,0.0,0.0] if scale < 1e-12 else [det/scale,nx/scale,ny/scale]
    inv_err = max(abs(float(numeric[6+i])-inv[i]) for i in range(3))
    max_invariant_error = max(max_invariant_error, inv_err)

if max_system_error > 2e-4:
    raise AssertionError(f"root-scaled Cramer feature error too large: {max_system_error}")
if max_invariant_error > 2e-5:
    raise AssertionError(f"Cramer invariant encoding mismatch: {max_invariant_error}")

polys = official_examples("polynomial_roots", 96, 0xA1653002)
true_residuals = []
perturbed_residuals = []
for e in polys:
    k, numeric, depth, fam, source = ns["encode"](e["eq"])
    if fam != ns["POLYNOMIAL"]:
        raise AssertionError(("poly family", source, fam))
    coeff = [float(x) for x in numeric[:6]]
    degree_feature = float(numeric[6])
    degree = max(i for i,c in enumerate(coeff) if abs(c) > 1e-10)
    if abs(degree_feature - degree/5.0) > 1e-6:
        raise AssertionError(("degree feature", source, degree, degree_feature))
    for root in e["roots"]:
        z = float(root)/ns["ROOT_SCALE"]
        true_residuals.append(abs(eval_poly(coeff,z)))
        # Three original x-units: large enough to create a useful residual signal,
        # small enough to stay in the same numeric neighborhood.
        perturbed_residuals.append(abs(eval_poly(coeff,z+0.03)))

true_sorted = sorted(true_residuals)
wrong_sorted = sorted(perturbed_residuals)
true_p95 = true_sorted[int(0.95*(len(true_sorted)-1))]
wrong_median = wrong_sorted[len(wrong_sorted)//2]
true_median = true_sorted[len(true_sorted)//2]
if true_p95 > 3e-3:
    raise AssertionError(f"true-root polynomial residual p95 too large: {true_p95}")
if wrong_median <= max(1e-5, true_median * 5.0):
    raise AssertionError(f"residual signal too weak: true median={true_median}, perturbed median={wrong_median}")

print("V3_CPU_MATH_CONTRACT_PASS")
print(f"official systems={len(systems)} max_cramer_solution_error={max_system_error:.3e} max_invariant_error={max_invariant_error:.3e}")
print(f"official polynomials={len(polys)} roots={len(true_residuals)} true_residual_p95={true_p95:.3e} perturbed_residual_median={wrong_median:.3e}")
print("project_synthetic_examples=0")
