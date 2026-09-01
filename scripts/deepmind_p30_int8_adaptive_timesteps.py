#!/usr/bin/env python3
"""Controlled adaptive-timestep / early-exit ablation for the adopted P30 INT8-QAT RSNN.

Baseline
--------
P30_INT8_QAT_T25: the adopted 30%-pruned, weight-only INT8 QAT RSNN using the
full fixed 25 recurrent timesteps.

Improvement under test
----------------------
Inference-only per-sample adaptive timesteps. A sample exits when its running
prediction average is stable for a fixed patience window. No labels, test
metrics, or future timesteps are consulted by the exit rule.

Three policies are declared before evaluation (no test-set threshold tuning):
- conservative: min 12 steps, patience 4, scaled-output delta <= 0.0015
- balanced:     min  8 steps, patience 3, scaled-output delta <= 0.0030
- aggressive:   min  6 steps, patience 2, scaled-output delta <= 0.0050

The model's output scale is 1/100 of the raw x,y targets, so those tolerances
correspond to 0.15, 0.30, and 0.50 raw coordinate units respectively.

Fairness contract
-----------------
- Same pinned DeepMind mathematics_dataset linear_2d banks as prior tests.
- Same five seeds, 300 epochs, P30 pruning schedule and INT8 QAT recipe.
- Same trained/exported INT8 weights are used by fixed-T25 and every adaptive
  policy for a seed. Only the inference stopping rule changes.
- Weight-only INT8 remains unchanged; activations/membrane/accumulators FP32.
- Report accuracy, robustness, average timesteps, exit rate, and theoretical
  recurrent/output weight-MAC reduction. No kernel-speedup claim is inferred
  from timestep reduction alone.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import pruned_snn_vs_mlp as arch
import deepmind_pruned_snn_stress as dm
import deepmind_p30_int8_ablation as qmod

DEVICE = torch.device("cpu")
BASELINE = "P30_INT8_QAT_T25"
POLICIES: Dict[str, Dict[str, float | int]] = {
    "P30_INT8_QAT_ADAPTIVE_CONSERVATIVE": {"min_steps": 12, "patience": 4, "abs_tol": 0.0015},
    "P30_INT8_QAT_ADAPTIVE_BALANCED": {"min_steps": 8, "patience": 3, "abs_tol": 0.0030},
    "P30_INT8_QAT_ADAPTIVE_AGGRESSIVE": {"min_steps": 6, "patience": 2, "abs_tol": 0.0050},
}
LABELS = (BASELINE, *POLICIES.keys())
METRICS = (
    "mae", "rmse", "median_sample_max_error", "p95_sample_max_error",
    "max_abs_error", "strict_0_1", "strict_0_5", "strict_1_0",
    "within_5_0", "relative_mae", "equation_residual_mean",
    "equation_residual_p95",
)


class AdaptiveInt8WeightRSNN(nn.Module):
    """Use an exported INT8-QAT RSNN with per-sample early exit."""

    def __init__(
        self,
        base: qmod.Int8WeightRSNN,
        *,
        min_steps: int,
        patience: int,
        abs_tol: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.min_steps = int(min_steps)
        self.patience = int(patience)
        self.abs_tol = float(abs_tol)
        self.time_steps = int(base.time_steps)
        if not (1 <= self.min_steps <= self.time_steps):
            raise ValueError("min_steps outside valid range")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.abs_tol < 0.0:
            raise ValueError("abs_tol must be >= 0")

    @torch.no_grad()
    def forward_with_steps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Dequantize the stored INT8 weights once per batch, exactly as the fixed
        # exported inference model does numerically.
        w_in = self.base._dequant("W_in")
        w_rec = self.base._dequant("W_rec")
        w_out = self.base._dequant("W_out")

        batch = x.shape[0]
        hidden = self.base.hidden_dim
        out_dim = self.base.out_dim
        mem = x.new_zeros(batch, hidden)
        spikes = x.new_zeros(batch, hidden)
        out_acc = x.new_zeros(batch, out_dim)
        prev_avg = x.new_zeros(batch, out_dim)
        final_out = x.new_zeros(batch, out_dim)
        stable_count = torch.zeros(batch, dtype=torch.int64, device=x.device)
        steps_used = torch.full((batch,), self.time_steps, dtype=torch.int64, device=x.device)
        done = torch.zeros(batch, dtype=torch.bool, device=x.device)

        synaptic_input = F.linear(x, w_in)
        for step in range(1, self.time_steps + 1):
            active_idx = torch.nonzero(~done, as_tuple=False).flatten()
            if active_idx.numel() == 0:
                break

            active_spikes = spikes[active_idx]
            recurrent_input = F.linear(active_spikes, w_rec)
            active_mem = self.base.decay * mem[active_idx] + synaptic_input[active_idx] + recurrent_input
            new_spikes = (active_mem - self.base.threshold >= 0.0).to(x.dtype)
            active_mem = active_mem - new_spikes * self.base.threshold
            active_out = out_acc[active_idx] + F.linear(new_spikes, w_out)
            running_avg = active_out / float(step)

            mem[active_idx] = active_mem
            spikes[active_idx] = new_spikes
            out_acc[active_idx] = active_out

            if step >= self.min_steps:
                delta = (running_avg - prev_avg[active_idx]).abs().amax(dim=1)
                stable_now = delta <= self.abs_tol
                old_counts = stable_count[active_idx]
                new_counts = torch.where(stable_now, old_counts + 1, torch.zeros_like(old_counts))
                stable_count[active_idx] = new_counts
                exit_local = new_counts >= self.patience
                if bool(exit_local.any()):
                    exiting_idx = active_idx[exit_local]
                    final_out[exiting_idx] = running_avg[exit_local]
                    steps_used[exiting_idx] = step
                    done[exiting_idx] = True

            prev_avg[active_idx] = running_avg

        remaining = torch.nonzero(~done, as_tuple=False).flatten()
        if remaining.numel() > 0:
            final_out[remaining] = out_acc[remaining] / float(self.time_steps)
            steps_used[remaining] = self.time_steps
        return final_out, steps_used

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred, _ = self.forward_with_steps(x)
        return pred


def metrics_from_pred(pred_scaled: torch.Tensor, bank: dm.Bank) -> Dict[str, float]:
    pred_raw = pred_scaled.cpu() * dm.TARGET_SCALE
    err = pred_raw - bank.y_raw
    abs_err = err.abs()
    sample_max = abs_err.amax(dim=1)
    x = bank.x
    r1 = x[:, 0] * pred_raw[:, 0] + x[:, 1] * pred_raw[:, 1] - x[:, 2]
    r2 = x[:, 3] * pred_raw[:, 0] + x[:, 4] * pred_raw[:, 1] - x[:, 5]
    residual = torch.maximum(r1.abs(), r2.abs())
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(torch.sqrt(torch.mean(err * err))),
        "median_sample_max_error": float(torch.quantile(sample_max, 0.50)),
        "p95_sample_max_error": float(torch.quantile(sample_max, 0.95)),
        "max_abs_error": float(abs_err.max()),
        "strict_0_1": float((sample_max <= 0.1).float().mean()),
        "strict_0_5": float((sample_max <= 0.5).float().mean()),
        "strict_1_0": float((sample_max <= 1.0).float().mean()),
        "within_5_0": float((sample_max <= 5.0).float().mean()),
        "relative_mae": float((abs_err / (bank.y_raw.abs() + 1.0)).mean()),
        "equation_residual_mean": float(residual.mean()),
        "equation_residual_p95": float(torch.quantile(residual, 0.95)),
    }


def work_report(base: qmod.Int8WeightRSNN, steps: torch.Tensor) -> Dict[str, float]:
    counts = base.quant_stats
    w_in = int(counts["W_in"]["active_count"])
    w_rec = int(counts["W_rec"]["active_count"])
    w_out = int(counts["W_out"]["active_count"])
    full_t = int(base.time_steps)
    mean_steps = float(steps.float().mean().item())
    fixed_work = float(w_in + full_t * (w_rec + w_out))
    adaptive_work = float(w_in + mean_steps * (w_rec + w_out))
    s = steps.float()
    return {
        "mean_steps": mean_steps,
        "median_steps": float(torch.quantile(s, 0.50).item()),
        "p90_steps": float(torch.quantile(s, 0.90).item()),
        "min_steps_used": float(s.min().item()),
        "max_steps_used": float(s.max().item()),
        "exit_before_t25_fraction": float((steps < full_t).float().mean().item()),
        "temporal_step_reduction_pct": 100.0 * (1.0 - mean_steps / full_t),
        "estimated_weight_mac_reduction_pct": 100.0 * (1.0 - adaptive_work / fixed_work),
        "fixed_active_weight_mac_units_per_sample": fixed_work,
        "adaptive_active_weight_mac_units_per_sample": adaptive_work,
    }


@torch.no_grad()
def evaluate_fixed(model: qmod.Int8WeightRSNN, bank: dm.Bank, batch_size: int = 512) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    outs: List[torch.Tensor] = []
    started = time.perf_counter()
    for start in range(0, len(bank.x), batch_size):
        outs.append(model(bank.x[start:start + batch_size].to(DEVICE)).cpu())
    elapsed = time.perf_counter() - started
    pred = torch.cat(outs, dim=0)
    steps = torch.full((len(bank.x),), model.time_steps, dtype=torch.int64)
    work = work_report(model, steps)
    work["wall_seconds"] = elapsed
    return metrics_from_pred(pred, bank), work


@torch.no_grad()
def evaluate_adaptive(model: AdaptiveInt8WeightRSNN, bank: dm.Bank, batch_size: int = 512) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    outs: List[torch.Tensor] = []
    step_rows: List[torch.Tensor] = []
    started = time.perf_counter()
    for start in range(0, len(bank.x), batch_size):
        pred, steps = model.forward_with_steps(bank.x[start:start + batch_size].to(DEVICE))
        outs.append(pred.cpu())
        step_rows.append(steps.cpu())
    elapsed = time.perf_counter() - started
    pred = torch.cat(outs, dim=0)
    steps = torch.cat(step_rows, dim=0)
    work = work_report(model.base, steps)
    work["wall_seconds"] = elapsed
    return metrics_from_pred(pred, bank), work


def aggregate(runs: List[Dict[str, object]], bank_names: Sequence[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    seeds = sorted({int(r["seed"]) for r in runs})
    for label in LABELS:
        rows = [r for r in runs if r["label"] == label]
        if len(rows) != len(seeds):
            raise AssertionError(f"missing runs for {label}: {len(rows)} != {len(seeds)}")
        entry: Dict[str, object] = {"banks": {}}
        for bank in bank_names:
            bank_out: Dict[str, object] = {"metrics": {}, "work": {}}
            for metric in METRICS:
                vals = [float(r["evaluations"][bank]["metrics"][metric]) for r in rows]
                bank_out["metrics"][metric] = {
                    "mean": mean(vals), "std": pstdev(vals), "min": min(vals), "max": max(vals)
                }
            for key in (
                "mean_steps", "median_steps", "p90_steps", "exit_before_t25_fraction",
                "temporal_step_reduction_pct", "estimated_weight_mac_reduction_pct", "wall_seconds",
            ):
                vals = [float(r["evaluations"][bank]["work"][key]) for r in rows]
                bank_out["work"][key] = {"mean": mean(vals), "std": pstdev(vals)}
            entry["banks"][bank] = bank_out
        out[label] = entry
    return out


def paired_compare(runs: List[Dict[str, object]], candidate: str, bank_names: Sequence[str]) -> Dict[str, object]:
    refs = {int(r["seed"]): r for r in runs if r["label"] == BASELINE}
    cands = {int(r["seed"]): r for r in runs if r["label"] == candidate}
    if set(refs) != set(cands):
        raise AssertionError("paired seed mismatch")
    out: Dict[str, object] = {}
    for bank in bank_names:
        mae_pct: List[float] = []
        strict1_pp: List[float] = []
        within5_pp: List[float] = []
        step_reduction: List[float] = []
        mac_reduction: List[float] = []
        wins = 0
        for seed in sorted(refs):
            a = refs[seed]["evaluations"][bank]
            b = cands[seed]["evaluations"][bank]
            am = float(a["metrics"]["mae"])
            bm = float(b["metrics"]["mae"])
            mae_pct.append(100.0 * (bm / am - 1.0))
            strict1_pp.append(100.0 * (float(b["metrics"]["strict_1_0"]) - float(a["metrics"]["strict_1_0"])))
            within5_pp.append(100.0 * (float(b["metrics"]["within_5_0"]) - float(a["metrics"]["within_5_0"])))
            step_reduction.append(float(b["work"]["temporal_step_reduction_pct"]))
            mac_reduction.append(float(b["work"]["estimated_weight_mac_reduction_pct"]))
            if bm < am:
                wins += 1
        out[bank] = {
            "mae_change_pct_mean": mean(mae_pct),
            "mae_change_pct_std": pstdev(mae_pct),
            "strict1_change_pp_mean": mean(strict1_pp),
            "within5_change_pp_mean": mean(within5_pp),
            "temporal_step_reduction_pct_mean": mean(step_reduction),
            "estimated_weight_mac_reduction_pct_mean": mean(mac_reduction),
            "mae_wins": wins,
            "mae_losses": len(refs) - wins,
        }
    return out


def preflight() -> None:
    qmod.set_seed(1234)
    source = arch.PrunedBrainSpikeNet()
    opt = torch.optim.AdamW(source.parameters(), lr=arch.LR, weight_decay=arch.WEIGHT_DECAY)
    source.prune_synapses(qmod.FINAL_SPARSITY, opt)
    base = qmod.Int8WeightRSNN(source).to(DEVICE)
    x = torch.randn(64, arch.IN_DIM)
    base.eval()
    with torch.no_grad():
        fixed = base(x)
        exact_policy = AdaptiveInt8WeightRSNN(base, min_steps=25, patience=1, abs_tol=1e9)
        adaptive, steps = exact_policy.forward_with_steps(x)
    max_diff = float((fixed - adaptive).abs().max().item())
    if max_diff > 1e-6 or not bool((steps == 25).all()):
        raise AssertionError(f"T25 equivalence failed diff={max_diff} steps={steps.unique().tolist()}")
    for label, cfg in POLICIES.items():
        model = AdaptiveInt8WeightRSNN(base, **cfg)
        pred, used = model.forward_with_steps(x)
        if pred.shape != fixed.shape or used.shape != (len(x),):
            raise AssertionError(f"shape failure {label}")
        if int(used.min()) < int(cfg["min_steps"]) or int(used.max()) > 25:
            raise AssertionError(f"step bounds failure {label}")
    print(f"ADAPTIVE_TIMESTEP_PREFLIGHT_PASS t25_equivalence_max_abs={max_diff:.3e}", flush=True)


def make_summary(agg: Dict[str, object], paired: Dict[str, object], storage: Dict[str, object]) -> str:
    official = "OFFICIAL_INTERPOLATE"
    lines = [
        "P30 INT8-QAT ADAPTIVE TIMESTEP ABLATION — DEEPMIND linear_2d",
        "",
        "Baseline: same P30 INT8-QAT model at fixed T=25.",
        "Adaptive policies are inference-only and predeclared; no test-set tuning.",
        f"Policies: {json.dumps(POLICIES, sort_keys=True)}",
        f"Storage unchanged: {json.dumps(storage, sort_keys=True)}",
        "",
        "OFFICIAL_INTERPOLATE:",
    ]
    for label in LABELS:
        row = agg[label]["banks"][official]
        m = row["metrics"]
        w = row["work"]
        lines.append(
            f"{label}: MAE={m['mae']['mean']:.8f} RMSE={m['rmse']['mean']:.8f} "
            f"strict1={100.0*m['strict_1_0']['mean']:.4f}% within5={100.0*m['within_5_0']['mean']:.4f}% "
            f"mean_steps={w['mean_steps']['mean']:.3f} step_reduction={w['temporal_step_reduction_pct']['mean']:.3f}% "
            f"est_weight_MAC_reduction={w['estimated_weight_mac_reduction_pct']['mean']:.3f}%"
        )
    lines += ["", "PAIRED VS FIXED T25:"]
    for label, banks in paired.items():
        for bank, row in banks.items():
            lines.append(
                f"{label} {bank}: MAE_delta={row['mae_change_pct_mean']:+.4f}% "
                f"strict1_delta={row['strict1_change_pp_mean']:+.4f}pp "
                f"within5_delta={row['within5_change_pp_mean']:+.4f}pp "
                f"steps_saved={row['temporal_step_reduction_pct_mean']:.3f}% "
                f"est_MAC_saved={row['estimated_weight_mac_reduction_pct_mean']:.3f}% "
                f"MAE_wins={row['mae_wins']}/5"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--eval-size", type=int, default=2048)
    parser.add_argument("--ill-pool-size", type=int, default=4096)
    parser.add_argument("--ill-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--output-dir", default="deepmind-p30-int8-adaptive-output")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    if args.preflight:
        preflight()
        return
    if args.epochs < qmod.QAT_START_EPOCH:
        raise ValueError("epochs do not cover pruning + QAT")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) < 3:
        raise ValueError("at least 3 seeds required")

    train, banks, manifest = qmod.build_banks(args)
    manifest["adaptive_timestep_contract"] = {
        "baseline_T": arch.TIME_STEPS,
        "policies": POLICIES,
        "selection": "predeclared before evaluation; no test-set tuning",
    }
    print("ADAPTIVE_FAIRNESS_CONTRACT same trained P30 INT8-QAT weights per seed; only inference stopping rule differs", flush=True)
    print(f"ADAPTIVE_POLICIES {json.dumps(POLICIES, sort_keys=True)}", flush=True)
    print("DATA_CONTRACT pinned google-deepmind/mathematics_dataset linear_2d; project synthetic data=0", flush=True)

    all_runs: List[Dict[str, object]] = []
    first_storage: Dict[str, object] | None = None
    for seed in seeds:
        qmod.set_seed(seed)
        qat = qmod.QATP30RSNN()
        qat, train_result = qmod.train_qat_variant(
            qat,
            train,
            banks["OFFICIAL_INTERPOLATE"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=seed,
        )
        exported = qmod.Int8WeightRSNN(qat).to(DEVICE)
        storage = exported.storage_report()
        if first_storage is None:
            first_storage = storage

        baseline_result: Dict[str, object] = {
            "seed": seed,
            "label": BASELINE,
            "training": train_result,
            "storage": storage,
            "evaluations": {},
        }
        for name, bank in banks.items():
            metrics, work = evaluate_fixed(exported, bank)
            baseline_result["evaluations"][name] = {"metrics": metrics, "work": work}
        all_runs.append(baseline_result)
        print(f"SEED_FINAL seed={seed} label={BASELINE} {json.dumps(baseline_result['evaluations'], sort_keys=True)}", flush=True)

        for label, cfg in POLICIES.items():
            adaptive = AdaptiveInt8WeightRSNN(exported, **cfg).to(DEVICE)
            result: Dict[str, object] = {
                "seed": seed,
                "label": label,
                "policy": cfg,
                "storage": storage,
                "evaluations": {},
            }
            for name, bank in banks.items():
                metrics, work = evaluate_adaptive(adaptive, bank)
                result["evaluations"][name] = {"metrics": metrics, "work": work}
            all_runs.append(result)
            print(f"SEED_FINAL seed={seed} label={label} {json.dumps(result['evaluations'], sort_keys=True)}", flush=True)

    if first_storage is None:
        raise AssertionError("missing storage")
    bank_names = list(banks.keys())
    agg = aggregate(all_runs, bank_names)
    paired = {label: paired_compare(all_runs, label, bank_names) for label in POLICIES}

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "P30_INT8_QAT_ADAPTIVE_TIMESTEPS",
        "device": str(DEVICE),
        "seeds": seeds,
        "epochs": args.epochs,
        "baseline_time_steps": arch.TIME_STEPS,
        "policies": POLICIES,
        "storage": first_storage,
        "runs": all_runs,
        "aggregate": agg,
        "paired_vs_fixed_t25": paired,
    }
    (output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    summary = make_summary(agg, paired, first_storage)
    (output / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary, flush=True)
    print("ADAPTIVE_TIMESTEP_EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
