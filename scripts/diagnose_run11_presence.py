#!/usr/bin/env python3
"""CPU-only diagnosis of Run #11 polynomial presence calibration.

No training. No synthetic data. Uses the exact MAI5-v3 checkpoint and official
DeepMind polynomial generators. Threshold selection uses independent calibration
banks; the frozen Run #11 audit banks are evaluation-only.
"""

import argparse
import copy
import json
import math
import pathlib
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=pathlib.Path, required=True)
    p.add_argument("--checkpoint", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    return p.parse_args()


def collect_bank(ns, examples, batch_size=128):
    torch = ns["torch"]
    np = ns["np"]
    model = ns["model"]
    device = ns["device"]
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            kinds = torch.tensor(np.stack([e["k"] for e in batch]), device=device, dtype=torch.long)
            numeric = torch.tensor(np.stack([e["n"] for e in batch]), device=device, dtype=torch.float32)
            depth = torch.tensor(np.stack([e["d"] for e in batch]), device=device, dtype=torch.float32)
            family = torch.tensor([e["f"] for e in batch], device=device, dtype=torch.long)
            out = model(kinds, numeric, depth, family)
            slots = (out[:, :5] * ns["ROOT_SCALE"]).cpu().numpy().astype(float)
            presence = torch.sigmoid(out[:, 5:10]).cpu().numpy().astype(float)
            state = torch.argmax(out[:, 10:14], dim=1).cpu().numpy().astype(int)
            for i, ex in enumerate(batch):
                rows.append({
                    "equation": ex["eq"],
                    "expected": [float(x) for x in ex["roots"]],
                    "expected_count": len(ex["roots"]),
                    "slots": [float(x) for x in slots[i]],
                    "presence": [float(x) for x in presence[i]],
                    "state": int(state[i]),
                    "expected_state": int(ex["state"]),
                })
    return rows


def metrics(rows, threshold, target_cap=300.0):
    sq = 0.0
    ae = 0.0
    value_count = 0
    within_one = 0
    count_ok = 0
    state_ok = 0
    missing = 0
    extras = 0
    confusion = Counter()
    for row in rows:
        expected = row["expected"]
        active = [i for i, p in enumerate(row["presence"]) if p >= threshold]
        predicted = [row["slots"][i] for i in active]
        true_count = len(expected)
        pred_count = len(predicted)
        count_ok += int(pred_count == true_count)
        state_ok += int(row["state"] == row["expected_state"])
        missing += max(0, true_count - pred_count)
        extras += max(0, pred_count - true_count)
        confusion[(true_count, pred_count)] += 1
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
                errs.append(target_cap)
        if errs:
            sq += sum(e * e for e in errs)
            ae += sum(errs)
            value_count += len(errs)
            within_one += sum(e <= 1.0 for e in errs)
    n = max(len(rows), 1)
    vc = max(value_count, 1)
    return {
        "threshold": round(float(threshold), 4),
        "root_count_accuracy": count_ok / n,
        "state_accuracy": state_ok / n,
        "missing_value_slots": int(missing),
        "extra_value_slots": int(extras),
        "mean_abs_count_error": (missing + extras) / n,
        "rmse": math.sqrt(sq / vc),
        "mae": ae / vc,
        "within_one_ratio": within_one / vc,
        "count_confusion": {f"{a}->{b}": c for (a, b), c in sorted(confusion.items())},
    }


def choose_threshold(interp_rows, extra_rows, thresholds):
    scored = []
    for t in thresholds:
        mi = metrics(interp_rows, t)
        me = metrics(extra_rows, t)
        key = (
            min(mi["root_count_accuracy"], me["root_count_accuracy"]),
            (mi["root_count_accuracy"] + me["root_count_accuracy"]) / 2.0,
            -(mi["mean_abs_count_error"] + me["mean_abs_count_error"]),
            (mi["within_one_ratio"] + me["within_one_ratio"]) / 2.0,
            -abs(t - 0.5),
        )
        scored.append((key, t, mi, me))
    return max(scored, key=lambda x: x[0]), scored


def best_for_bank(rows, thresholds):
    vals = [(metrics(rows, t), t) for t in thresholds]
    return max(
        vals,
        key=lambda x: (
            x[0]["root_count_accuracy"],
            -x[0]["mean_abs_count_error"],
            x[0]["within_one_ratio"],
            -abs(x[1] - 0.5),
        ),
    )[0]


def summarize_presence(rows):
    # Diagnostic only: sorted presence probabilities by true root count. This does
    # not assume a slot-to-root identity and therefore remains permutation-safe.
    by_count = {}
    for count in range(1, 6):
        selected = [sorted(r["presence"], reverse=True) for r in rows if r["expected_count"] == count]
        if not selected:
            continue
        columns = list(zip(*selected))
        by_count[str(count)] = {
            "examples": len(selected),
            "median_sorted_presence": [float(__import__("statistics").median(c)) for c in columns],
        }
    return by_count


def main():
    args = parse_args()
    root = args.repo_root.resolve()
    checkpoint = args.checkpoint.resolve()
    out_path = args.output.resolve()

    # Import the audited bank builder, not a project-local synthetic generator.
    import importlib.util
    audit_path = root / "colab/generalization_audit.py"
    spec = importlib.util.spec_from_file_location("dm_family_audit", audit_path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)

    ns = audit.load_runtime(root)
    torch = ns["torch"]
    if torch.cuda.is_available():
        raise RuntimeError("CPU_ONLY_DIAGNOSIS_REFUSES_CUDA")
    if str(ns["device"]) != "cpu":
        raise RuntimeError(f"Expected CPU runtime, got {ns['device']}")

    ns["load_mai5"](str(checkpoint))
    print(f"Loaded checkpoint={checkpoint} adam_step={ns['adam_step']} device={ns['device']}")

    frozen_i_spec = copy.deepcopy(audit.BANK_SPECS["interpolate_polynomial"])
    frozen_e_spec = copy.deepcopy(audit.BANK_SPECS["extrapolate_polynomial"])
    cal_i_spec = copy.deepcopy(frozen_i_spec)
    cal_e_spec = copy.deepcopy(frozen_e_spec)
    cal_i_spec["seed"] = 0xC411B001
    cal_e_spec["seed"] = 0xC411B002
    # Keep calibration size equal to audit size so root-count distributions are comparable.
    cal_i_spec["count"] = 256
    cal_e_spec["count"] = 256

    print("Building official DeepMind calibration banks...")
    cal_i = audit.build_official_bank(ns, cal_i_spec)
    cal_e = audit.build_official_bank(ns, cal_e_spec)
    print("Building frozen Run11 audit banks...")
    frozen_i = audit.build_official_bank(ns, frozen_i_spec)
    frozen_e = audit.build_official_bank(ns, frozen_e_spec)

    cal_i_rows = collect_bank(ns, cal_i)
    cal_e_rows = collect_bank(ns, cal_e)
    frozen_i_rows = collect_bank(ns, frozen_i)
    frozen_e_rows = collect_bank(ns, frozen_e)

    thresholds = [x / 100.0 for x in range(5, 96)]
    baseline = {
        "interpolate": metrics(frozen_i_rows, 0.5),
        "extrapolate": metrics(frozen_e_rows, 0.5),
    }
    # Sanity-check against Run11 gate numbers; fail rather than calibrate a different bank/runtime.
    if abs(baseline["interpolate"]["root_count_accuracy"] - 0.5234375) > 1e-9:
        raise RuntimeError(f"Frozen interpolate reproduction mismatch: {baseline['interpolate']['root_count_accuracy']}")
    if abs(baseline["extrapolate"]["root_count_accuracy"] - 0.5234375) > 1e-9:
        raise RuntimeError(f"Frozen extrapolate reproduction mismatch: {baseline['extrapolate']['root_count_accuracy']}")

    chosen, calibration_sweep = choose_threshold(cal_i_rows, cal_e_rows, thresholds)
    _, chosen_t, chosen_cal_i, chosen_cal_e = chosen
    selected_eval = {
        "interpolate": metrics(frozen_i_rows, chosen_t),
        "extrapolate": metrics(frozen_e_rows, chosen_t),
    }
    oracle = {
        "interpolate": best_for_bank(frozen_i_rows, thresholds),
        "extrapolate": best_for_bank(frozen_e_rows, thresholds),
    }

    interp_target = float(frozen_i_spec["min_count_accuracy"])
    extra_target = float(frozen_e_spec["min_count_accuracy"])
    threshold_only_can_pass_count_gate = (
        selected_eval["interpolate"]["root_count_accuracy"] >= interp_target
        and selected_eval["extrapolate"]["root_count_accuracy"] >= extra_target
    )
    oracle_can_pass_both = (
        oracle["interpolate"]["root_count_accuracy"] >= interp_target
        and oracle["extrapolate"]["root_count_accuracy"] >= extra_target
    )

    result = {
        "schema": "RUN11_POLYNOMIAL_PRESENCE_CALIBRATION_V1",
        "cpu_only": True,
        "training_performed": False,
        "project_synthetic_examples": 0,
        "checkpoint": checkpoint.name,
        "adam_step": int(ns["adam_step"]),
        "bank_sizes": {
            "calibration_interpolate": len(cal_i_rows),
            "calibration_extrapolate": len(cal_e_rows),
            "frozen_interpolate": len(frozen_i_rows),
            "frozen_extrapolate": len(frozen_e_rows),
        },
        "gate_count_targets": {"interpolate": interp_target, "extrapolate": extra_target},
        "baseline_threshold_0_5": baseline,
        "selected_threshold": float(chosen_t),
        "selected_on_independent_calibration": {
            "interpolate": chosen_cal_i,
            "extrapolate": chosen_cal_e,
        },
        "selected_threshold_on_frozen_audit": selected_eval,
        "oracle_best_on_frozen_audit_diagnostic_only": oracle,
        "threshold_only_can_pass_count_gate": bool(threshold_only_can_pass_count_gate),
        "oracle_threshold_can_pass_both_count_gates": bool(oracle_can_pass_both),
        "presence_shape_frozen_interpolate": summarize_presence(frozen_i_rows),
        "presence_shape_frozen_extrapolate": summarize_presence(frozen_e_rows),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("BASELINE 0.50", json.dumps(baseline, sort_keys=True))
    print(f"SELECTED_THRESHOLD={chosen_t:.2f}")
    print("SELECTED_FROZEN", json.dumps(selected_eval, sort_keys=True))
    print("ORACLE_FROZEN", json.dumps(oracle, sort_keys=True))
    print(f"THRESHOLD_ONLY_CAN_PASS_COUNT_GATE={threshold_only_can_pass_count_gate}")
    print(f"ORACLE_THRESHOLD_CAN_PASS_BOTH_COUNT_GATES={oracle_can_pass_both}")


if __name__ == "__main__":
    main()
