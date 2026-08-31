#!/usr/bin/env python3
"""DeepMind RSNN pruning sweep: 0/10/20/30/40/50% sparsity.

All variants share the exact RSNN architecture, optimizer, DeepMind banks,
initial weights, epoch count and deterministic batch order for each seed.
The only experimental variable is final magnitude-pruning sparsity.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

import pruned_snn_vs_mlp as arch
import deepmind_pruned_snn_stress as dm

DEVICE = torch.device("cpu")
SPARSITIES = (0.00, 0.10, 0.20, 0.30, 0.40, 0.50)
METRICS = (
    "mae", "rmse", "median_sample_max_error", "p95_sample_max_error",
    "max_abs_error", "strict_0_1", "strict_0_5", "strict_1_0",
    "within_5_0", "relative_mae", "equation_residual_mean",
    "equation_residual_p95",
)


def label_for(sparsity: float) -> str:
    return f"P{int(round(100.0 * sparsity)):02d}"


def pruning_target(epoch: int, final_sparsity: float) -> float | None:
    if final_sparsity <= 0.0:
        return None
    if epoch < arch.PRUNE_START or epoch > arch.PRUNE_END:
        return None
    if (epoch - arch.PRUNE_START) % arch.PRUNE_EVERY != 0:
        return None
    span = max(1, arch.PRUNE_END - arch.PRUNE_START)
    progress = (epoch - arch.PRUNE_START) / span
    return float(final_sparsity * progress)


def train_variant(
    model: arch.PrunedBrainSpikeNet,
    train: dm.Bank,
    monitor: dm.Bank,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    final_sparsity: float,
) -> Tuple[arch.PrunedBrainSpikeNet, Dict[str, object]]:
    label = label_for(final_sparsity)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arch.LR, weight_decay=arch.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=arch.ETA_MIN
    )
    criterion = nn.SmoothL1Loss(beta=1.0)
    checkpoints = {1, 50, 100, 150, 200, 250, epochs}
    checkpoints = {e for e in checkpoints if 1 <= e <= epochs}
    history: List[Dict[str, object]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        target = pruning_target(epoch, final_sparsity)
        if target is not None:
            report = model.prune_synapses(target, optimizer)
            print(
                f"PRUNE seed={seed} label={label} epoch={epoch} "
                f"target={target:.4f} actual={json.dumps(report, sort_keys=True)}",
                flush=True,
            )

        model.train()
        running = 0.0
        seen = 0
        for ids in arch.deterministic_batches(len(train.x), batch_size, seed, epoch):
            bx = train.x[ids].to(DEVICE)
            by = train.y_scaled[ids].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = criterion(pred, by)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss seed={seed} label={label} epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            model.zero_pruned_gradients_()
            optimizer.step()
            model.apply_masks_()
            running += float(loss.detach()) * len(ids)
            seen += len(ids)
        scheduler.step()

        if epoch in checkpoints:
            row: Dict[str, object] = {
                "epoch": epoch,
                "loss": running / max(seen, 1),
                "lr": optimizer.param_groups[0]["lr"],
                "official_interpolate": dm.evaluate_raw(model, monitor),
                "sparsity": model.sparsity_report(),
            }
            history.append(row)
            print(f"CHECKPOINT seed={seed} label={label} {json.dumps(row, sort_keys=True)}", flush=True)

    elapsed = time.perf_counter() - started
    report = model.sparsity_report()
    for key in ("W_in", "W_rec", "W_out", "overall"):
        if abs(float(report[key]) - final_sparsity) > 0.002:
            raise AssertionError(
                f"final sparsity mismatch seed={seed} label={label} {key}={report[key]} expected={final_sparsity}"
            )

    return model, {
        "seed": seed,
        "label": label,
        "final_sparsity": final_sparsity,
        "params": arch.count_parameters(model),
        "active_weights": int(round(arch.count_parameters(model) * (1.0 - final_sparsity))),
        "train_seconds": elapsed,
        "sparsity": report,
        "history": history,
    }


def aggregate(runs: List[Dict[str, object]], bank_names: Sequence[str]) -> Dict[str, object]:
    out: Dict[str, object] = {"by_sparsity": {}, "paired_vs_dense": {}}
    seeds = sorted({int(r["seed"]) for r in runs})

    for sparsity in SPARSITIES:
        label = label_for(sparsity)
        rows = [r for r in runs if r["label"] == label]
        if len(rows) != len(seeds):
            raise AssertionError(f"missing runs for {label}: {len(rows)} != {len(seeds)}")
        entry: Dict[str, object] = {
            "final_sparsity": sparsity,
            "active_weights": int(rows[0]["active_weights"]),
            "train_seconds": {
                "mean": mean(float(r["train_seconds"]) for r in rows),
                "std": pstdev(float(r["train_seconds"]) for r in rows) if len(rows) > 1 else 0.0,
            },
            "banks": {},
        }
        for bank in bank_names:
            bank_out: Dict[str, object] = {}
            for metric in METRICS:
                vals = [float(r["evaluations"][bank][metric]) for r in rows]
                bank_out[metric] = {
                    "mean": mean(vals),
                    "std": pstdev(vals) if len(vals) > 1 else 0.0,
                    "min": min(vals),
                    "max": max(vals),
                }
            entry["banks"][bank] = bank_out
        out["by_sparsity"][label] = entry

    dense = {int(r["seed"]): r for r in runs if r["label"] == "P00"}
    for sparsity in SPARSITIES[1:]:
        label = label_for(sparsity)
        rows = {int(r["seed"]): r for r in runs if r["label"] == label}
        paired_bank: Dict[str, object] = {}
        for bank in bank_names:
            mae_pct = []
            strict1_pp = []
            within5_pp = []
            residual_pct = []
            wins_mae = 0
            for seed in seeds:
                d = dense[seed]["evaluations"][bank]
                p = rows[seed]["evaluations"][bank]
                d_mae = float(d["mae"])
                p_mae = float(p["mae"])
                mae_pct.append(100.0 * (p_mae / d_mae - 1.0))
                strict1_pp.append(100.0 * (float(p["strict_1_0"]) - float(d["strict_1_0"])))
                within5_pp.append(100.0 * (float(p["within_5_0"]) - float(d["within_5_0"])))
                d_res = float(d["equation_residual_mean"])
                p_res = float(p["equation_residual_mean"])
                residual_pct.append(100.0 * (p_res / d_res - 1.0))
                if p_mae < d_mae:
                    wins_mae += 1
            paired_bank[bank] = {
                "mae_change_pct_mean": mean(mae_pct),
                "mae_change_pct_std": pstdev(mae_pct),
                "strict1_change_pp_mean": mean(strict1_pp),
                "strict1_change_pp_std": pstdev(strict1_pp),
                "within5_change_pp_mean": mean(within5_pp),
                "residual_change_pct_mean": mean(residual_pct),
                "mae_wins_vs_dense": wins_mae,
                "mae_losses_vs_dense": len(seeds) - wins_mae,
            }
        out["paired_vs_dense"][label] = paired_bank

    official = "OFFICIAL_INTERPOLATE"
    ranking = []
    dense_official_mae = float(out["by_sparsity"]["P00"]["banks"][official]["mae"]["mean"])
    dense_official_s1 = float(out["by_sparsity"]["P00"]["banks"][official]["strict_1_0"]["mean"])
    for sparsity in SPARSITIES:
        label = label_for(sparsity)
        row = out["by_sparsity"][label]
        off = row["banks"][official]
        ranking.append({
            "label": label,
            "sparsity": sparsity,
            "active_weights": row["active_weights"],
            "official_mae": off["mae"]["mean"],
            "official_mae_change_pct": 100.0 * (float(off["mae"]["mean"]) / dense_official_mae - 1.0),
            "official_strict1": off["strict_1_0"]["mean"],
            "official_strict1_change_pp": 100.0 * (float(off["strict_1_0"]["mean"]) - dense_official_s1),
        })
    out["official_tradeoff_table"] = ranking
    return out


def preflight(dm_algebra) -> None:
    seen: set = set()
    train = dm.build_bank(dm_algebra, name="SWEEP_PREFLIGHT_TRAIN", kind="train", count=64, seed=27001, global_seen=seen)
    test = dm.build_bank(dm_algebra, name="SWEEP_PREFLIGHT_TEST", kind="interpolate", count=48, seed=28001, global_seen=seen)

    initial = None
    for sparsity in SPARSITIES:
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        model = arch.PrunedBrainSpikeNet()
        weights = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith("W_")}
        if initial is None:
            initial = weights
        else:
            for name in initial:
                if not torch.equal(initial[name], weights[name]):
                    raise AssertionError(f"initialization mismatch {label_for(sparsity)} {name}")
        if sparsity > 0:
            opt = torch.optim.AdamW(model.parameters(), lr=arch.LR, weight_decay=arch.WEIGHT_DECAY)
            report = model.prune_synapses(sparsity, opt)
            if abs(float(report["overall"]) - sparsity) > 0.001:
                raise AssertionError(report)

    torch.manual_seed(4321)
    smoke = arch.PrunedBrainSpikeNet()
    smoke, result = train_variant(
        smoke, train, test, epochs=2, batch_size=32, seed=4321, final_sparsity=0.0
    )
    if abs(float(result["sparsity"]["overall"])) > 1e-12:
        raise AssertionError("dense smoke unexpectedly sparse")
    print("PRUNING_SWEEP_PREFLIGHT_PASS ratios=0,10,20,30,40,50 identical_initialization=true", flush=True)


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
    parser.add_argument("--output-dir", default="deepmind-rsnn-pruning-sweep-output")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    dm_algebra = dm.prepare_deepmind()
    if args.preflight:
        preflight(dm_algebra)
        return

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) < 3:
        raise ValueError("sweep requires at least 3 seeds")
    if args.epochs < arch.PRUNE_END:
        raise ValueError("epochs must cover full pruning schedule")

    seen: set = set()
    train = dm.build_bank(dm_algebra, name="TRAIN_DEEPMIND", kind="train", count=args.train_size, seed=7301, global_seen=seen)
    interp = dm.build_bank(dm_algebra, name="OFFICIAL_INTERPOLATE", kind="interpolate", count=args.eval_size, seed=8301, global_seen=seen)
    stress10 = dm.build_bank(dm_algebra, name="STRESS_ENTROPY_10", kind="entropy10", count=args.eval_size, seed=9301, global_seen=seen)
    stress12 = dm.build_bank(dm_algebra, name="STRESS_ENTROPY_12", kind="entropy12", count=args.eval_size, seed=10301, global_seen=seen)
    ill_pool = dm.build_bank(dm_algebra, name="ILL_POOL_ENTROPY_12", kind="entropy12", count=args.ill_pool_size, seed=11301, global_seen=seen)
    ill_ids = torch.argsort(ill_pool.condition, descending=True)[: args.ill_size]
    ill = dm.subset_bank(ill_pool, ill_ids, "ILL_CONDITIONED")
    banks = {interp.name: interp, stress10.name: stress10, stress12.name: stress12, ill.name: ill}

    params = arch.count_parameters(arch.PrunedBrainSpikeNet())
    if params != 26880:
        raise AssertionError(f"unexpected RSNN parameter count {params}")
    active = {label_for(s): int(round(params * (1.0 - s))) for s in SPARSITIES}
    print(f"SWEEP_PARAMETER_CONTRACT params_each={params} active_weights={json.dumps(active, sort_keys=True)}", flush=True)
    print("SWEEP_CONTRACT same RSNN architecture/optimizer/data/batches/seeds/initialization; only final pruning ratio differs", flush=True)
    print("DATA_CONTRACT official google-deepmind/mathematics_dataset linear_2d only; project synthetic data=0", flush=True)

    all_runs: List[Dict[str, object]] = []
    for seed in seeds:
        reference_initial = None
        for sparsity in SPARSITIES:
            label = label_for(sparsity)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = arch.PrunedBrainSpikeNet()
            initial = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith("W_")}
            if reference_initial is None:
                reference_initial = initial
            else:
                for name in reference_initial:
                    if not torch.equal(reference_initial[name], initial[name]):
                        raise AssertionError(f"seed={seed} initialization mismatch label={label} param={name}")

            model, result = train_variant(
                model,
                train,
                interp,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=seed,
                final_sparsity=sparsity,
            )
            result["evaluations"] = {name: dm.evaluate_raw(model, bank) for name, bank in banks.items()}
            print(f"SEED_FINAL seed={seed} label={label} {json.dumps(result['evaluations'], sort_keys=True)}", flush=True)
            all_runs.append(result)
            del model

    bank_names = list(banks.keys())
    summary = aggregate(all_runs, bank_names)
    manifest = {
        "deepmind_repo": dm.DM_REPO,
        "deepmind_commit": dm.DM_COMMIT,
        "seeds": seeds,
        "epochs": args.epochs,
        "sparsities": list(SPARSITIES),
        "params_each": params,
        "active_weights": active,
        "project_synthetic_examples": 0,
        "banks": {
            train.name: dm.bank_manifest(train),
            **{name: dm.bank_manifest(bank) for name, bank in banks.items()},
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"manifest": manifest, "runs": all_runs, "aggregate": summary}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    lines = [
        "DeepMind RSNN pruning sweet-spot sweep",
        f"DeepMind commit={dm.DM_COMMIT}",
        f"Seeds={seeds}",
        f"Train={len(train.x)} OfficialInterp={len(interp.x)} Stress10={len(stress10.x)} Stress12={len(stress12.x)} IllConditioned={len(ill.x)}",
        f"Params each={params}",
        "Sparsities=0%,10%,20%,30%,40%,50%",
        "Project synthetic examples=0",
        "",
        "OFFICIAL TRADEOFF:",
    ]
    for row in summary["official_tradeoff_table"]:
        lines.append(
            f"  {row['label']}: sparsity={100*row['sparsity']:.0f}% active={row['active_weights']} "
            f"MAE={float(row['official_mae']):.6f} deltaMAE={float(row['official_mae_change_pct']):+.2f}% "
            f"strict1={float(row['official_strict1']):.6f} deltaStrict1={float(row['official_strict1_change_pp']):+.2f}pp"
        )
    for bank in bank_names:
        lines.append("")
        lines.append(bank)
        for sparsity in SPARSITIES:
            label = label_for(sparsity)
            a = summary["by_sparsity"][label]["banks"][bank]
            lines.append(
                f"  {label}: MAE={a['mae']['mean']:.6f}±{a['mae']['std']:.6f} "
                f"strict1={a['strict_1_0']['mean']:.6f} within5={a['within_5_0']['mean']:.6f} "
                f"residual={a['equation_residual_mean']['mean']:.6f}"
            )
        for sparsity in SPARSITIES[1:]:
            label = label_for(sparsity)
            p = summary["paired_vs_dense"][label][bank]
            lines.append(
                f"    {label} vs P00 paired: MAE={p['mae_change_pct_mean']:+.2f}%±{p['mae_change_pct_std']:.2f}% "
                f"strict1={p['strict1_change_pp_mean']:+.2f}pp wins={p['mae_wins_vs_dense']}/5"
            )
    for sparsity in SPARSITIES:
        label = label_for(sparsity)
        t = summary["by_sparsity"][label]["train_seconds"]
        lines.append(f"TRAIN_SECONDS {label}: {t['mean']:.3f}±{t['std']:.3f}")

    (out_dir / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    print("FINAL_PRUNING_SWEEP", json.dumps(summary, sort_keys=True), flush=True)
    print("RSNN_PRUNING_SWEEP_DEEPMIND_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
