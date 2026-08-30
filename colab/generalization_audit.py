# Math AI v5 — independent generalization / anti-memorization audit
# Runs AFTER training against the selected MAI5 checkpoint.
# It does not prove that memorization is impossible, but it deliberately tests
# distributions and equation forms that were not fed to the optimizer.

import argparse
import json
import math
import pathlib
import random
import re
import time


def replace_setting(text, name, value_repr):
    pattern = rf"(?m)^{re.escape(name)}\s*=\s*.*$"
    updated, count = re.subn(pattern, f"{name} = {value_repr}", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not patch trainer setting: {name}")
    return updated


def load_runtime(root: pathlib.Path):
    trainer_path = root / "colab/train_v5_deepmind.py"
    src = trainer_path.read_text()
    src = replace_setting(src, "RESUME_FROM_MAI5", "False")
    src = replace_setting(src, "AUTO_DOWNLOAD_AT_END", "False")
    prefix = src.split("# ========================= TRAIN =========================", 1)[0]
    ns = {"__name__": "mai5_generalization_audit"}
    exec(compile(prefix, str(trainer_path), "exec"), ns)
    return ns


def target_magnitude(example):
    vals = list(example.get("roots", ())) + list(example.get("system", ()))
    return max((abs(float(v)) for v in vals), default=0.0)


def build_strict_ood(ns, count=256):
    # Training synthetic range is <=100. This bank only accepts finite targets
    # whose actual requested solution magnitude is >=135, creating a true
    # coefficient/solution-range separation rather than merely using a larger RNG.
    rng = random.Random(0x5A17D00D)
    result = []
    seen = set()
    attempts = 0
    while len(result) < count and attempts < count * 1200:
        attempts += 1
        ex = ns["synthetic"](rng, max_abs=240)
        if ex["state"] != ns["FINITE"]:
            continue
        if target_magnitude(ex) < 135.0:
            continue
        key = ex["eq"]
        if key in seen:
            continue
        seen.add(key)
        result.append(ex)
    if len(result) < max(64, count // 2):
        raise RuntimeError(f"Could only create {len(result)} strict OOD examples")
    return result


def build_fresh_iid(ns, count=256):
    # Independent seed; this stream is never used by training.
    rng = random.Random(0x1A2B3C4D)
    result = []
    seen = set()
    while len(result) < count:
        ex = ns["synthetic"](rng, max_abs=100)
        if ex["eq"] in seen:
            continue
        seen.add(ex["eq"])
        result.append(ex)
    return result


def build_deepmind_split(ns, split="interpolate", count=192):
    dm_algebra = ns["dm_algebra"]
    old_modules = ns["dm_modules"]
    old_names = list(ns["DM_NAMES"])
    try:
        if split == "interpolate":
            ns["dm_modules"] = dm_algebra.test()
            ns["DM_NAMES"] = ["linear_1d", "linear_2d", "polynomial_roots"]
        elif split == "extrapolate":
            ns["dm_modules"] = dm_algebra.test_extra()
            ns["DM_NAMES"] = ["polynomial_roots_big"]
        else:
            raise ValueError(split)
        rng = random.Random(0xD33F0001 if split == "interpolate" else 0xD33F0002)
        result = []
        seen = set()
        attempts = 0
        while len(result) < count and attempts < count * 100:
            attempts += 1
            ex = ns["deepmind_example"](rng)
            if ex["eq"] in seen:
                continue
            seen.add(ex["eq"])
            result.append(ex)
        if len(result) < max(48, count // 2):
            raise RuntimeError(f"DeepMind {split} produced only {len(result)} parsed examples")
        return result
    finally:
        ns["dm_modules"] = old_modules
        ns["DM_NAMES"] = old_names


def equivalent_add_constant(eq, amount=37):
    # This transformation is intentionally different from the side-swap used
    # during training: add the same unseen constant to BOTH sides.
    pieces = []
    for part in eq.split(";"):
        left, right = part.split("=", 1)
        pieces.append(f"({left})+{amount}=({right})+{amount}")
    return ";".join(pieces)


def predict_raw(ns, model, equation):
    torch = ns["torch"]
    np = ns["np"]
    k, n, d, fam, src = ns["encode"](equation)
    device = ns["device"]
    kt = torch.tensor(k[None, :], device=device, dtype=torch.long)
    nt = torch.tensor(n[None, :], device=device, dtype=torch.float32)
    dt = torch.tensor(d[None, :], device=device, dtype=torch.float32)
    ft = torch.tensor([fam], device=device, dtype=torch.long)
    with torch.no_grad():
        out = model(kt, nt, dt, ft)[0]
        slots = (out[:5] * ns["ROOT_SCALE"]).detach().cpu().numpy().astype(float)
        presence = torch.sigmoid(out[5:10]).detach().cpu().numpy().astype(float)
        state_probs = torch.softmax(out[10:14], dim=0).detach().cpu().numpy().astype(float)
    return {
        "family": int(fam),
        "slots": slots,
        "presence": presence,
        "state_probs": state_probs,
        "state": int(np.argmax(state_probs)),
    }


def eval_examples(ns, model, examples):
    np = ns["np"]
    finite = ns["FINITE"]
    system_family = ns["SYSTEM"]
    sq = 0.0
    ae = 0.0
    count = 0
    within = 0
    state_ok = 0
    missing = 0

    model.eval()
    for ex in examples:
        p = predict_raw(ns, model, ex["eq"])
        if p["state"] == ex["state"]:
            state_ok += 1
        if ex["state"] != finite:
            continue
        if ex["f"] == system_family:
            expected = np.asarray(ex["system"], dtype=float)
            predicted = p["slots"][:len(expected)]
            errors = np.abs(predicted - expected)
        else:
            expected = np.asarray(ex["roots"], dtype=float)
            active = p["presence"] >= 0.5
            predicted = p["slots"][active]
            used = set()
            errs = []
            for value in expected:
                candidates = [(abs(float(v) - float(value)), j) for j, v in enumerate(predicted) if j not in used]
                if candidates:
                    er, j = min(candidates)
                    used.add(j)
                    errs.append(er)
                else:
                    errs.append(float(ns["ROOT_SCALE"]))
                    missing += 1
            errors = np.asarray(errs, dtype=float)
        if len(errors):
            sq += float((errors ** 2).sum())
            ae += float(errors.sum())
            count += len(errors)
            within += int((errors <= 1.0).sum())

    return {
        "examples": len(examples),
        "value_count": int(count),
        "rmse": math.sqrt(sq / max(count, 1)),
        "mae": ae / max(count, 1),
        "within_one_ratio": within / max(count, 1),
        "state_accuracy": state_ok / max(len(examples), 1),
        "missing_value_slots": int(missing),
    }


def consistency_audit(ns, model, examples, limit=128):
    np = ns["np"]
    slot_deltas = []
    presence_deltas = []
    state_kl = []
    family_mismatch = 0
    checked = 0
    for ex in examples[:limit]:
        transformed = equivalent_add_constant(ex["eq"], 37)
        try:
            a = predict_raw(ns, model, ex["eq"])
            b = predict_raw(ns, model, transformed)
        except Exception:
            continue
        checked += 1
        if a["family"] != b["family"]:
            family_mismatch += 1
            continue
        slot_deltas.append(float(np.mean(np.abs(a["slots"] - b["slots"]))))
        presence_deltas.append(float(np.mean(np.abs(a["presence"] - b["presence"]))))
        pa = np.clip(a["state_probs"], 1e-9, 1.0)
        pb = np.clip(b["state_probs"], 1e-9, 1.0)
        state_kl.append(float(np.sum(pa * np.log(pa / pb))))
    return {
        "checked": checked,
        "family_mismatch": family_mismatch,
        "mean_slot_delta": float(np.mean(slot_deltas)) if slot_deltas else float("inf"),
        "mean_presence_delta": float(np.mean(presence_deltas)) if presence_deltas else float("inf"),
        "mean_state_kl": float(np.mean(state_kl)) if state_kl else float("inf"),
        "transformation": "add +37 to both sides; not used as the training equivalence transform",
    }


def improvement(random_metric, trained_metric):
    base = max(float(random_metric["rmse"]), 1e-9)
    return (base - float(trained_metric["rmse"])) / base


def verdict(audit):
    gains = audit["improvement_vs_random"]
    trained = audit["trained"]
    consistency = audit["equivalence_consistency"]
    passed = 0
    reasons = []

    for name in ("fresh_iid", "strict_ood", "deepmind_interpolate", "deepmind_extrapolate"):
        gain = gains[name]
        if gain >= 0.20:
            passed += 1
        else:
            reasons.append(f"{name}: only {gain*100:.1f}% RMSE improvement over random")

    iid = trained["fresh_iid"]["rmse"]
    ood = trained["strict_ood"]["rmse"]
    if iid > 0 and ood / iid > 3.0:
        reasons.append(f"large OOD gap: strict-OOD RMSE is {ood/iid:.2f}x fresh-IID")

    if consistency["mean_slot_delta"] > max(5.0, trained["fresh_iid"]["mae"] * 2.0):
        reasons.append(f"equivalent-equation instability: slot delta {consistency['mean_slot_delta']:.3f}")

    if passed >= 4 and not reasons:
        level = "PASS"
    elif passed >= 3:
        level = "WARN"
    else:
        level = "FAIL"
    return level, reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/content/MathAI-v5-factory")
    parser.add_argument("--model", default="/content/math_ai_v5_best.mai5")
    parser.add_argument("--output", default="/content/generalization_audit.json")
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    model_path = pathlib.Path(args.model)
    output = pathlib.Path(args.output)
    if not model_path.is_file():
        raise RuntimeError(f"Missing trained model: {model_path}")

    print("\n=== GENERALIZATION / ANTI-MEMORIZATION AUDIT ===")
    started = time.time()
    ns = load_runtime(root)
    torch = ns["torch"]

    fresh = build_fresh_iid(ns, 256)
    strict_ood = build_strict_ood(ns, 256)
    dm_interp = build_deepmind_split(ns, "interpolate", 192)
    dm_extra = build_deepmind_split(ns, "extrapolate", 128)

    # Random, never-trained network is the baseline.
    random_model = ns["MAI5"]().to(ns["device"])
    random_metrics = {
        "fresh_iid": eval_examples(ns, random_model, fresh),
        "strict_ood": eval_examples(ns, random_model, strict_ood),
        "deepmind_interpolate": eval_examples(ns, random_model, dm_interp),
        "deepmind_extrapolate": eval_examples(ns, random_model, dm_extra),
    }

    trained_model = ns["model"]
    ns["load_mai5"](str(model_path))
    trained_metrics = {
        "fresh_iid": eval_examples(ns, trained_model, fresh),
        "strict_ood": eval_examples(ns, trained_model, strict_ood),
        "deepmind_interpolate": eval_examples(ns, trained_model, dm_interp),
        "deepmind_extrapolate": eval_examples(ns, trained_model, dm_extra),
    }
    consistency = consistency_audit(ns, trained_model, fresh, 128)

    gains = {name: improvement(random_metrics[name], trained_metrics[name]) for name in trained_metrics}
    audit = {
        "purpose": "Evidence that learning generalizes beyond memorized training instances/templates",
        "important_limit": "No finite test can mathematically prove that a neural network never memorizes. These disjoint tests are strong empirical controls.",
        "sets": {
            "fresh_iid": "independent RNG stream, same nominal range, never used by optimizer",
            "strict_ood": "finite targets with |solution| >= 135 while training synthetic range <= 100",
            "deepmind_interpolate": "official DeepMind algebra test() split",
            "deepmind_extrapolate": "official DeepMind algebra test_extra() polynomial_roots_big split",
            "equivalence": "+37 added to both sides, a transformation not used as the training consistency pair",
        },
        "random_baseline": random_metrics,
        "trained": trained_metrics,
        "improvement_vs_random": gains,
        "equivalence_consistency": consistency,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    level, reasons = verdict(audit)
    audit["verdict"] = level
    audit["warnings"] = reasons
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    print("\nRandom -> trained RMSE improvement:")
    for name, gain in gains.items():
        print(f"  {name:22s}: {gain*100:7.2f}%   trained RMSE={trained_metrics[name]['rmse']:.4f}")
    print(f"Equivalent-form mean slot delta: {consistency['mean_slot_delta']:.4f}")
    print("GENERALIZATION VERDICT:", level)
    for reason in reasons:
        print("  WARNING:", reason)
    print("Audit JSON:", output)


if __name__ == "__main__":
    main()
