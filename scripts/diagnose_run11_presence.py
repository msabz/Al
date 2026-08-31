#!/usr/bin/env python3
"""CPU-only diagnosis of Run #11 polynomial presence calibration.

No training and no project synthetic data. The frozen evaluation banks are rebuilt
in the exact order used by Run #11 before any independent calibration bank is
created. Threshold selection never uses the frozen audit banks.
"""

import argparse
import copy
import importlib.util
import json
import math
import pathlib
import random
import statistics
from collections import Counter


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=pathlib.Path, required=True)
    p.add_argument("--checkpoint", type=pathlib.Path, required=True)
    p.add_argument("--run11-audit", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    return p.parse_args()


def collect_bank(ns, examples, batch_size=128):
    torch, np, model, device = ns["torch"], ns["np"], ns["model"], ns["device"]
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
    sq = ae = 0.0
    value_count = within_one = count_ok = state_ok = missing = extras = 0
    confusion = Counter()
    for row in rows:
        expected = row["expected"]
        active = [i for i, p in enumerate(row["presence"]) if p >= threshold]
        predicted = [row["slots"][i] for i in active]
        true_count, pred_count = len(expected), len(predicted)
        count_ok += int(pred_count == true_count)
        state_ok += int(row["state"] == row["expected_state"])
        missing += max(0, true_count - pred_count)
        extras += max(0, pred_count - true_count)
        confusion[(true_count, pred_count)] += 1
        used, errs = set(), []
        for value in expected:
            candidates = [(abs(float(v) - float(value)), j) for j, v in enumerate(predicted) if j not in used]
            if candidates:
                err, j = min(candidates)
                used.add(j)
                errs.append(err)
            else:
                errs.append(target_cap)
        sq += sum(e * e for e in errs)
        ae += sum(errs)
        value_count += len(errs)
        within_one += sum(e <= 1.0 for e in errs)
    n, vc = max(len(rows), 1), max(value_count, 1)
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
        mi, me = metrics(interp_rows, t), metrics(extra_rows, t)
        key = (
            min(mi["root_count_accuracy"], me["root_count_accuracy"]),
            (mi["root_count_accuracy"] + me["root_count_accuracy"]) / 2.0,
            -(mi["mean_abs_count_error"] + me["mean_abs_count_error"]),
            (mi["within_one_ratio"] + me["within_one_ratio"]) / 2.0,
            -abs(t - 0.5),
        )
        scored.append((key, t, mi, me))
    return max(scored, key=lambda x: x[0])


def best_for_bank(rows, thresholds):
    vals = [metrics(rows, t) for t in thresholds]
    return max(vals, key=lambda m: (
        m["root_count_accuracy"], -m["mean_abs_count_error"],
        m["within_one_ratio"], -abs(m["threshold"] - 0.5)))


def presence_shape(rows):
    out = {}
    for count in range(1, 6):
        vals = [sorted(r["presence"], reverse=True) for r in rows if r["expected_count"] == count]
        if vals:
            cols = list(zip(*vals))
            out[str(count)] = {
                "examples": len(vals),
                "median_sorted_presence": [float(statistics.median(c)) for c in cols],
            }
    return out


def reseed_official(ns, seed):
    random.seed(int(seed))
    ns["np"].random.seed(int(seed) & 0xFFFFFFFF)
    ns["torch"].manual_seed(int(seed))


def verify_frozen_bank_against_run11(audit, name, spec, examples, ref_row):
    meta = audit.bank_metadata(spec, examples)
    ref_meta = ref_row["metadata"]
    for key in ("examples", "value_count", "max_abs_target", "seed"):
        if meta[key] != ref_meta[key]:
            raise RuntimeError(f"Frozen bank metadata mismatch {name}.{key}: {meta[key]} != {ref_meta[key]}")
    equations = {e["eq"] for e in examples}
    reference_worst = [w["equation"] for w in ref_row["trained"].get("worst", [])]
    missing = [eq for eq in reference_worst if eq not in equations]
    if missing:
        raise RuntimeError(f"Frozen bank equation mismatch {name}: {len(missing)} Run11 worst equations absent")
    print(f"RUN11_FROZEN_BANK_REPRODUCED {name} values={meta['value_count']} worst_members={len(reference_worst)}")


def main():
    args = args_parser()
    root, checkpoint, out_path = args.repo_root.resolve(), args.checkpoint.resolve(), args.output.resolve()
    reference = json.loads(args.run11_audit.resolve().read_text())
    if reference.get("schema") != "DEEPMIND_ONLY_PER_FAMILY_V2":
        raise RuntimeError("Unexpected Run11 audit schema")

    audit_path = root / "colab/generalization_audit.py"
    spec = importlib.util.spec_from_file_location("dm_family_audit", audit_path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    ns = audit.load_runtime(root)
    torch = ns["torch"]
    if torch.cuda.is_available() or str(ns["device"]) != "cpu":
        raise RuntimeError("CPU_ONLY_DIAGNOSIS_REFUSES_CUDA")

    # Critical: reproduce Run11's generator state by building ALL frozen banks in
    # the exact insertion order used by generalization_audit.py.
    print("Building exact Run11 frozen banks in original order...")
    frozen = {}
    for name, bank_spec in audit.BANK_SPECS.items():
        examples = audit.build_official_bank(ns, copy.deepcopy(bank_spec))
        frozen[name] = examples
        verify_frozen_bank_against_run11(audit, name, bank_spec, examples, reference["banks"][name])

    ns["load_mai5"](str(checkpoint))
    print(f"Loaded checkpoint={checkpoint} adam_step={ns['adam_step']} device={ns['device']}")

    frozen_i = frozen["interpolate_polynomial"]
    frozen_e = frozen["extrapolate_polynomial"]
    frozen_i_rows, frozen_e_rows = collect_bank(ns, frozen_i), collect_bank(ns, frozen_e)

    # Independent calibration banks: explicit global RNG reseed plus the adapter's
    # own Random(seed). These are not used as release/audit evidence.
    cal_i_spec = copy.deepcopy(audit.BANK_SPECS["interpolate_polynomial"])
    cal_e_spec = copy.deepcopy(audit.BANK_SPECS["extrapolate_polynomial"])
    cal_i_spec.update(seed=0xC411B001, count=128)
    cal_e_spec.update(seed=0xC411B002, count=128)
    print("Building independent official-DeepMind calibration banks...")
    reseed_official(ns, 0xC411B001)
    cal_i = audit.build_official_bank(ns, cal_i_spec)
    reseed_official(ns, 0xC411B002)
    cal_e = audit.build_official_bank(ns, cal_e_spec)
    cal_i_rows, cal_e_rows = collect_bank(ns, cal_i), collect_bank(ns, cal_e)

    thresholds = [x / 100.0 for x in range(5, 96)]
    baseline = {"interpolate": metrics(frozen_i_rows, 0.5), "extrapolate": metrics(frozen_e_rows, 0.5)}
    ref_counts = {
        "interpolate": reference["banks"]["interpolate_polynomial"]["trained"]["root_count_accuracy"],
        "extrapolate": reference["banks"]["extrapolate_polynomial"]["trained"]["root_count_accuracy"],
    }
    # GPU-vs-CPU arithmetic can flip logits extremely close to zero. Bank identity
    # is validated above; keep the metric delta visible rather than pretending exactness.
    baseline_count_delta = {k: baseline[k]["root_count_accuracy"] - ref_counts[k] for k in baseline}

    chosen = choose_threshold(cal_i_rows, cal_e_rows, thresholds)
    _, chosen_t, chosen_cal_i, chosen_cal_e = chosen
    selected_eval = {"interpolate": metrics(frozen_i_rows, chosen_t), "extrapolate": metrics(frozen_e_rows, chosen_t)}
    oracle = {"interpolate": best_for_bank(frozen_i_rows, thresholds), "extrapolate": best_for_bank(frozen_e_rows, thresholds)}

    i_target = float(audit.BANK_SPECS["interpolate_polynomial"]["min_count_accuracy"])
    e_target = float(audit.BANK_SPECS["extrapolate_polynomial"]["min_count_accuracy"])
    selected_pass = selected_eval["interpolate"]["root_count_accuracy"] >= i_target and selected_eval["extrapolate"]["root_count_accuracy"] >= e_target
    oracle_pass = oracle["interpolate"]["root_count_accuracy"] >= i_target and oracle["extrapolate"]["root_count_accuracy"] >= e_target

    result = {
        "schema": "RUN11_POLYNOMIAL_PRESENCE_CALIBRATION_V2",
        "cpu_only": True,
        "training_performed": False,
        "project_synthetic_examples": 0,
        "checkpoint": checkpoint.name,
        "adam_step": int(ns["adam_step"]),
        "frozen_bank_contract": "Run11 order + metadata + membership of every Run11 stored worst-case equation",
        "bank_sizes": {"calibration_interpolate": 128, "calibration_extrapolate": 128, "frozen_interpolate": 256, "frozen_extrapolate": 256},
        "gate_count_targets": {"interpolate": i_target, "extrapolate": e_target},
        "run11_gpu_reference_count_accuracy": ref_counts,
        "baseline_threshold_0_5_cpu": baseline,
        "baseline_cpu_minus_run11_gpu_count_accuracy": baseline_count_delta,
        "selected_threshold": float(chosen_t),
        "selected_on_independent_calibration": {"interpolate": chosen_cal_i, "extrapolate": chosen_cal_e},
        "selected_threshold_on_frozen_audit_cpu": selected_eval,
        "oracle_best_on_frozen_audit_cpu_diagnostic_only": oracle,
        "threshold_only_can_pass_count_gate": bool(selected_pass),
        "oracle_threshold_can_pass_both_count_gates": bool(oracle_pass),
        "presence_shape_frozen_interpolate": presence_shape(frozen_i_rows),
        "presence_shape_frozen_extrapolate": presence_shape(frozen_e_rows),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("BASELINE_CPU_0.50", json.dumps(baseline, sort_keys=True))
    print("RUN11_GPU_REFERENCE_COUNT", json.dumps(ref_counts, sort_keys=True))
    print(f"SELECTED_THRESHOLD={chosen_t:.2f}")
    print("SELECTED_FROZEN_CPU", json.dumps(selected_eval, sort_keys=True))
    print("ORACLE_FROZEN_CPU", json.dumps(oracle, sort_keys=True))
    print(f"THRESHOLD_ONLY_CAN_PASS_COUNT_GATE={selected_pass}")
    print(f"ORACLE_THRESHOLD_CAN_PASS_BOTH_COUNT_GATES={oracle_pass}")


if __name__ == "__main__":
    main()
