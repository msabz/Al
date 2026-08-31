#!/usr/bin/env python3
"""Math AI v5 — DeepMind-only per-family generalization audit.

This audit uses only the official DeepMind Mathematics Dataset generators:
  * algebra.test(): linear_1d, linear_2d, polynomial_roots
  * algebra.test_extra(): polynomial_roots_big
No project synthetic generator and no project-generated equivalence transforms are used.

The audit is deliberately per-family. It does not compare raw RMSE between
interpolation and extrapolation distributions, because their target scales differ.
Instead, every bank is compared with the median of multiple fresh random networks,
and an independent absolute accuracy floor is also required.
"""

import argparse
import json
import math
import pathlib
import random
import re
import statistics
import time

AUDIT_SCHEMA = "DEEPMIND_ONLY_PER_FAMILY_V2"
TARGET_CAP = 300.0
RANDOM_BASELINE_SEEDS = (7301, 7302, 7303, 7304, 7305)

BANK_SPECS = {
    "interpolate_linear_1d": {
        "split": "interpolate", "module": "linear_1d", "count": 256,
        "seed": 0xA1654101, "family": "LINEAR",
        "min_rmse_gain": 0.80, "min_mae_gain": 0.80, "min_within_one": 0.50,
        "min_count_accuracy": 0.95,
    },
    "interpolate_linear_2d": {
        "split": "interpolate", "module": "linear_2d", "count": 256,
        "seed": 0xA1654102, "family": "SYSTEM",
        "min_rmse_gain": 0.50, "min_mae_gain": 0.55, "min_within_one": 0.40,
        "min_count_accuracy": 0.95,
    },
    "interpolate_polynomial": {
        "split": "interpolate", "module": "polynomial_roots", "count": 256,
        "seed": 0xA1654103, "family": "POLYNOMIAL",
        "min_rmse_gain": 0.40, "min_mae_gain": 0.50, "min_within_one": 0.25,
        "min_count_accuracy": 0.75,
    },
    "extrapolate_polynomial": {
        "split": "extrapolate", "module": "polynomial_roots_big", "count": 256,
        "seed": 0xA1654104, "family": "POLYNOMIAL",
        "min_rmse_gain": 0.30, "min_mae_gain": 0.40, "min_within_one": 0.15,
        "min_count_accuracy": 0.65,
    },
}


def replace_setting(text, name, value_repr):
    pattern = rf"(?m)^{re.escape(name)}\s*=\s*.*$"
    updated, count = re.subn(pattern, f"{name} = {value_repr}", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not patch trainer setting: {name}")
    return updated


def load_runtime(root: pathlib.Path):
    trainer_path = root / "colab/train_v5_deepmind.py"
    if not trainer_path.is_file():
        raise RuntimeError(f"Missing trainer: {trainer_path}")
    runtime_dir = root / ".deepmind-audit-runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    src = trainer_path.read_text()
    # Make the definitions portable to Kaggle/GitHub/Termux-style runners.
    src = src.replace("/content", runtime_dir.as_posix())
    src = replace_setting(src, "RESUME_FROM_MAI5", "False")
    src = replace_setting(src, "AUTO_DOWNLOAD_AT_END", "False")
    prefix = src.split("# ========================= TRAIN =========================", 1)[0]
    ns = {"__name__": "mai5_deepmind_family_audit", "__file__": str(trainer_path)}
    exec(compile(prefix, str(trainer_path), "exec"), ns)
    return ns


def target_magnitude(example):
    values = list(example.get("roots", ())) + list(example.get("system", ()))
    return max((abs(float(v)) for v in values), default=0.0)


def expected_family_id(ns, name):
    return {
        "LINEAR": ns["LINEAR"],
        "POLYNOMIAL": ns["POLYNOMIAL"],
        "SYSTEM": ns["SYSTEM"],
    }[name]


def build_official_bank(ns, spec):
    dm_algebra = ns["dm_algebra"]
    old_modules = ns["dm_modules"]
    old_names = list(ns["DM_NAMES"])
    old_synthetic = ns["synthetic"]
    expected_family = expected_family_id(ns, spec["family"])

    def project_synthetic_forbidden(*args, **kwargs):
        raise RuntimeError("PROJECT_SYNTHETIC_FORBIDDEN_IN_AUDIT")

    try:
        ns["synthetic"] = project_synthetic_forbidden
        if spec["split"] == "interpolate":
            ns["dm_modules"] = dm_algebra.test()
        elif spec["split"] == "extrapolate":
            ns["dm_modules"] = dm_algebra.test_extra()
        else:
            raise ValueError(spec["split"])
        ns["DM_NAMES"] = [spec["module"]]

        rng = random.Random(int(spec["seed"]))
        result = []
        seen = set()
        attempts = 0
        max_attempts = int(spec["count"]) * 2500
        while len(result) < int(spec["count"]) and attempts < max_attempts:
            attempts += 1
            try:
                ex = ns["deepmind_example"](rng, allow_synthetic_fallback=False)
            except Exception:
                continue
            if ex["f"] != expected_family:
                raise RuntimeError(
                    f"DeepMind adapter family mismatch for {spec['module']}: "
                    f"expected={spec['family']} actual={ex['f']} equation={ex['eq']}"
                )
            if ex["state"] != ns["FINITE"]:
                continue
            if target_magnitude(ex) > TARGET_CAP:
                continue
            key = ex["eq"]
            if key in seen:
                continue
            seen.add(key)
            result.append(ex)

        if len(result) != int(spec["count"]):
            raise RuntimeError(
                f"Official DeepMind bank {spec['split']}/{spec['module']} produced "
                f"only {len(result)}/{spec['count']} usable unique examples after {attempts} attempts"
            )
        return result
    finally:
        ns["synthetic"] = old_synthetic
        ns["dm_modules"] = old_modules
        ns["DM_NAMES"] = old_names


def predict_raw(ns, model, equation):
    torch = ns["torch"]
    np = ns["np"]
    k, n, d, fam, normalized = ns["encode"](equation)
    device = ns["device"]
    with torch.no_grad():
        kinds = torch.tensor(k[None, :], device=device, dtype=torch.long)
        numeric = torch.tensor(n[None, :], device=device, dtype=torch.float32)
        depth = torch.tensor(d[None, :], device=device, dtype=torch.float32)
        out = model(kinds, numeric, depth, torch.tensor([fam], device=device, dtype=torch.long))[0]
        slots = (out[:5] * ns["ROOT_SCALE"]).detach().cpu().numpy().astype(float)
        if fam == ns["POLYNOMIAL"]:
            count_probs = torch.softmax(out[5:10], dim=0).detach().cpu().numpy().astype(float)
            active = ns["polynomial_active_indices"](out, numeric[0]).detach().cpu().numpy().astype(int).tolist()
            presence = np.zeros(5, dtype=float)
            presence[active] = 1.0
        else:
            count_probs = np.asarray([], dtype=float)
            presence = torch.sigmoid(out[5:10]).detach().cpu().numpy().astype(float)
        state_probs = torch.softmax(out[10:14], dim=0).detach().cpu().numpy().astype(float)
    return {
        "family": int(fam),
        "state": int(np.argmax(state_probs)),
        "slots": slots,
        "presence": presence,
        "root_count_probs": count_probs,
        "state_probs": state_probs,
        "normalized": normalized,
    }

def eval_examples(ns, model, examples):
    np = ns["np"]
    system_family = ns["SYSTEM"]
    sq = 0.0
    ae = 0.0
    value_count = 0
    within_one = 0
    state_ok = 0
    count_ok = 0
    missing = 0
    extras = 0
    worst = []

    model.eval()
    for ex in examples:
        p = predict_raw(ns, model, ex["eq"])
        state_ok += int(p["state"] == ex["state"])
        if ex["f"] == system_family:
            expected = np.asarray(ex["system"], dtype=float)
            predicted = p["slots"][:len(expected)]
            errors = np.abs(predicted - expected)
            count_ok += int(len(expected) == 2)
        else:
            expected = np.asarray(ex["roots"], dtype=float)
            active_indices = [i for i, prob in enumerate(p["presence"]) if prob >= 0.5]
            predicted = np.asarray([p["slots"][i] for i in active_indices], dtype=float)
            count_ok += int(len(predicted) == len(expected))
            extras += max(0, len(predicted) - len(expected))
            used = set()
            errs = []
            for value in expected:
                candidates = [
                    (abs(float(v) - float(value)), j)
                    for j, v in enumerate(predicted)
                    if j not in used
                ]
                if candidates:
                    err, j = min(candidates)
                    used.add(j)
                    errs.append(err)
                else:
                    errs.append(TARGET_CAP)
                    missing += 1
            errors = np.asarray(errs, dtype=float)

        if len(errors):
            sq += float((errors ** 2).sum())
            ae += float(errors.sum())
            value_count += len(errors)
            within_one += int((errors <= 1.0).sum())
            max_err = float(errors.max())
            worst.append({
                "max_abs_error": max_err,
                "equation": ex["eq"],
                "expected": [float(x) for x in expected],
                "predicted": [float(x) for x in predicted],
            })

    worst.sort(key=lambda row: row["max_abs_error"], reverse=True)
    n = max(len(examples), 1)
    vc = max(value_count, 1)
    return {
        "examples": len(examples),
        "value_count": int(value_count),
        "rmse": math.sqrt(sq / vc),
        "mae": ae / vc,
        "within_one_ratio": within_one / vc,
        "state_accuracy": state_ok / n,
        "root_count_accuracy": count_ok / n,
        "missing_value_slots": int(missing),
        "extra_value_slots": int(extras),
        "worst": worst[:8],
    }


def median_metrics(rows):
    scalar_keys = (
        "rmse", "mae", "within_one_ratio", "state_accuracy", "root_count_accuracy",
        "missing_value_slots", "extra_value_slots",
    )
    out = {}
    for key in scalar_keys:
        out[key] = float(statistics.median(float(r[key]) for r in rows))
    out["runs"] = len(rows)
    return out


def fresh_random_baseline(ns, examples):
    torch = ns["torch"]
    rows = []
    for seed in RANDOM_BASELINE_SEEDS:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        random_model = ns["MAI5"]().to(ns["device"])
        rows.append(eval_examples(ns, random_model, examples))
    return {"median": median_metrics(rows), "runs": rows, "seeds": list(RANDOM_BASELINE_SEEDS)}


def improvement(base, trained, key):
    denom = max(float(base[key]), 1e-9)
    return (float(base[key]) - float(trained[key])) / denom


def evaluate_gate(name, spec, trained, baseline):
    gains = {
        "rmse_gain": improvement(baseline, trained, "rmse"),
        "mae_gain": improvement(baseline, trained, "mae"),
    }
    reasons = []
    checks = [
        (gains["rmse_gain"] >= spec["min_rmse_gain"], f"RMSE gain {gains['rmse_gain']*100:.1f}% < {spec['min_rmse_gain']*100:.1f}%"),
        (gains["mae_gain"] >= spec["min_mae_gain"], f"MAE gain {gains['mae_gain']*100:.1f}% < {spec['min_mae_gain']*100:.1f}%"),
        (trained["within_one_ratio"] >= spec["min_within_one"], f"±1 accuracy {trained['within_one_ratio']*100:.1f}% < {spec['min_within_one']*100:.1f}%"),
        (trained["root_count_accuracy"] >= spec["min_count_accuracy"], f"root-count accuracy {trained['root_count_accuracy']*100:.1f}% < {spec['min_count_accuracy']*100:.1f}%"),
        (trained["state_accuracy"] >= 0.95, f"state accuracy {trained['state_accuracy']*100:.1f}% < 95.0%"),
    ]
    for ok, reason in checks:
        if not ok:
            reasons.append(f"{name}: {reason}")
    return gains, reasons


def bank_metadata(spec, examples):
    values = []
    for ex in examples:
        values.extend(float(v) for v in ex.get("roots", ()))
        values.extend(float(v) for v in ex.get("system", ()))
    return {
        "split": spec["split"],
        "module": spec["module"],
        "expected_family": spec["family"],
        "examples": len(examples),
        "value_count": len(values),
        "max_abs_target": max((abs(v) for v in values), default=0.0),
        "target_cap": TARGET_CAP,
        "seed": int(spec["seed"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/content/MathAI-v5-factory")
    parser.add_argument("--model", default="/content/math_ai_v5_best.mai5")
    parser.add_argument("--output", default="/content/generalization_audit.json")
    parser.add_argument("--validate-banks-only", action="store_true")
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    output = pathlib.Path(args.output)
    started = time.time()
    print("\n=== DEEPMIND-ONLY PER-FAMILY GENERALIZATION AUDIT ===", flush=True)
    ns = load_runtime(root)

    banks = {}
    for name, spec in BANK_SPECS.items():
        examples = build_official_bank(ns, spec)
        banks[name] = examples
        meta = bank_metadata(spec, examples)
        print(
            f"BANK_OK {name}: {meta['examples']} examples, {meta['value_count']} values, "
            f"max|target|={meta['max_abs_target']:.3f}",
            flush=True,
        )

    if args.validate_banks_only:
        payload = {
            "schema": AUDIT_SCHEMA,
            "verdict": "BANKS_OK",
            "data_contract": "official DeepMind test/test_extra only; project synthetic forbidden",
            "banks": {name: bank_metadata(BANK_SPECS[name], ex) for name, ex in banks.items()},
            "elapsed_seconds": round(time.time() - started, 3),
        }
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print("DEEPMIND_AUDIT_BANKS_VALID", flush=True)
        return

    model_path = pathlib.Path(args.model)
    if not model_path.is_file():
        raise RuntimeError(f"Missing trained model: {model_path}")

    trained_model = ns["model"]
    ns["load_mai5"](str(model_path))

    results = {}
    warnings = []
    for name, examples in banks.items():
        baseline = fresh_random_baseline(ns, examples)
        trained = eval_examples(ns, trained_model, examples)
        gains, reasons = evaluate_gate(name, BANK_SPECS[name], trained, baseline["median"])
        warnings.extend(reasons)
        results[name] = {
            "metadata": bank_metadata(BANK_SPECS[name], examples),
            "requirements": {k: v for k, v in BANK_SPECS[name].items() if k.startswith("min_")},
            "random_baseline": baseline,
            "trained": trained,
            "gains": gains,
            "passed": not reasons,
        }
        print(
            f"{name}: RMSE={trained['rmse']:.4f}, MAE={trained['mae']:.4f}, "
            f"±1={trained['within_one_ratio']*100:.2f}%, count={trained['root_count_accuracy']*100:.2f}%, "
            f"RMSE-gain={gains['rmse_gain']*100:.2f}%",
            flush=True,
        )

    verdict = "PASS" if not warnings and all(row["passed"] for row in results.values()) else "FAIL"
    audit = {
        "schema": AUDIT_SCHEMA,
        "verdict": verdict,
        "purpose": "Per-family empirical generalization and practical numerical quality gate",
        "data_contract": {
            "source": "official DeepMind Mathematics Dataset generators",
            "interpolate": "algebra.test()",
            "extrapolate": "algebra.test_extra()",
            "project_synthetic_examples": 0,
            "project_generated_equivalence_examples": 0,
            "target_support_cap": TARGET_CAP,
        },
        "method": {
            "comparison": "each family vs median of five independently initialized fresh MAI5 networks",
            "cross_distribution_raw_rmse_ratio": "not used",
            "reason": "interpolation and extrapolation have different target distributions/scales",
            "random_baseline_seeds": list(RANDOM_BASELINE_SEEDS),
        },
        "banks": results,
        "warnings": warnings,
        "elapsed_seconds": round(time.time() - started, 3),
        # Explicit compatibility marker for old orchestration. No equivalence test is fabricated.
        "equivalence_consistency": {
            "checked": 0,
            "mean_slot_delta": 0.0,
            "disabled_reason": "DeepMind-only contract; no project-generated transforms",
        },
    }
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print("GENERALIZATION_VERDICT=" + verdict, flush=True)
    if warnings:
        for warning in warnings:
            print("GATE_FAIL", warning, flush=True)


if __name__ == "__main__":
    main()
