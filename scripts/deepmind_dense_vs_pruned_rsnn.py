#!/usr/bin/env python3
"""Ablation benchmark: Dense RSNN vs 30%-pruned RSNN vs parameter-matched MLP.

This deliberately reuses the exact DeepMind data/parser/evaluation pipeline from
`deepmind_pruned_snn_stress.py` so the only SNN difference is pruning.
For a given seed DENSE_RSNN and PRUNED_RSNN start from bit-identical weights.
No project-synthetic examples are used.
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
LABELS = ("DENSE_RSNN", "PRUNED_RSNN", "ORDINARY_MLP")


def train_variant(
    model: nn.Module,
    train: dm.Bank,
    monitor: dm.Bank,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    label: str,
    enable_pruning: bool,
) -> Tuple[nn.Module, Dict[str, object]]:
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arch.LR, weight_decay=arch.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=arch.ETA_MIN
    )
    criterion = nn.SmoothL1Loss(beta=1.0)
    checkpoints = {1, 25, 50, 100, 150, 200, 250, epochs}
    checkpoints = {e for e in checkpoints if 1 <= e <= epochs}
    history = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if enable_pruning:
            target = arch.pruning_target_for_epoch(epoch)
            if target is not None:
                report = model.prune_synapses(target, optimizer)
                print(
                    f"PRUNE seed={seed} label={label} epoch={epoch} target={target:.4f} actual={report}",
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
            if isinstance(model, arch.PrunedBrainSpikeNet):
                # Dense SNN masks remain all ones, so this is a no-op there.
                model.zero_pruned_gradients_()
            optimizer.step()
            if isinstance(model, arch.PrunedBrainSpikeNet):
                model.apply_masks_()
            running += float(loss.detach()) * len(ids)
            seen += len(ids)
        scheduler.step()

        if epoch in checkpoints:
            metrics = dm.evaluate_raw(model, monitor)
            row: Dict[str, object] = {
                "epoch": epoch,
                "loss": running / max(seen, 1),
                "lr": optimizer.param_groups[0]["lr"],
                "official_interpolate": metrics,
            }
            if isinstance(model, arch.PrunedBrainSpikeNet):
                row["sparsity"] = model.sparsity_report()
            history.append(row)
            print(f"CHECKPOINT seed={seed} label={label} {json.dumps(row, sort_keys=True)}", flush=True)

    elapsed = time.perf_counter() - started
    result: Dict[str, object] = {
        "seed": seed,
        "label": label,
        "params": arch.count_parameters(model),
        "train_seconds": elapsed,
        "history": history,
    }
    if isinstance(model, arch.PrunedBrainSpikeNet):
        report = model.sparsity_report()
        result["sparsity"] = report
        expected = arch.FINAL_SPARSITY if enable_pruning else 0.0
        for key in ("W_in", "W_rec", "W_out"):
            if abs(report[key] - expected) > 0.002:
                raise AssertionError(f"{label} sparsity mismatch {key}={report[key]} expected={expected}")
    return model, result


def aggregate(all_runs: List[Dict[str, object]], bank_names: Sequence[str]) -> Dict[str, object]:
    metrics = [
        "mae", "rmse", "median_sample_max_error", "p95_sample_max_error",
        "max_abs_error", "strict_0_1", "strict_0_5", "strict_1_0",
        "within_5_0", "relative_mae", "equation_residual_mean",
        "equation_residual_p95",
    ]
    out: Dict[str, object] = {}
    for label in LABELS:
        out[label] = {}
        runs = [r for r in all_runs if r["label"] == label]
        for bank in bank_names:
            out[label][bank] = {}
            for metric in metrics:
                vals = [float(r["evaluations"][bank][metric]) for r in runs]
                out[label][bank][metric] = {
                    "mean": mean(vals),
                    "std": pstdev(vals) if len(vals) > 1 else 0.0,
                    "min": min(vals),
                    "max": max(vals),
                }
        out[label]["train_seconds"] = {
            "mean": mean(float(r["train_seconds"]) for r in runs),
            "std": pstdev(float(r["train_seconds"]) for r in runs) if len(runs) > 1 else 0.0,
        }

    seeds = sorted({int(r["seed"]) for r in all_runs})
    wins: Dict[str, Dict[str, int]] = {}
    dense_vs_pruned: Dict[str, Dict[str, float]] = {}
    for bank in bank_names:
        counts = {label: 0 for label in LABELS}
        for seed in seeds:
            rows = [r for r in all_runs if int(r["seed"]) == seed]
            # Primary ablation decision: lower raw MAE. Tie-break by higher strict1.
            winner = min(
                rows,
                key=lambda r: (
                    float(r["evaluations"][bank]["mae"]),
                    -float(r["evaluations"][bank]["strict_1_0"]),
                ),
            )
            counts[str(winner["label"])] += 1
        wins[bank] = counts

        d = out["DENSE_RSNN"][bank]
        p = out["PRUNED_RSNN"][bank]
        dense_vs_pruned[bank] = {
            "dense_mae": float(d["mae"]["mean"]),
            "pruned_mae": float(p["mae"]["mean"]),
            "pruned_mae_change_pct_vs_dense": 100.0 * (float(p["mae"]["mean"]) / float(d["mae"]["mean"]) - 1.0),
            "dense_strict_1": float(d["strict_1_0"]["mean"]),
            "pruned_strict_1": float(p["strict_1_0"]["mean"]),
            "strict_1_point_change_pruned_minus_dense": 100.0 * (float(p["strict_1_0"]["mean"]) - float(d["strict_1_0"]["mean"])),
        }
    out["seed_wins_by_mae"] = wins
    out["dense_vs_pruned"] = dense_vs_pruned
    return out


def preflight(dm_algebra) -> None:
    seen: set = set()
    train = dm.build_bank(dm_algebra, name="ABLATION_PREFLIGHT_TRAIN", kind="train", count=64, seed=17001, global_seen=seen)
    test = dm.build_bank(dm_algebra, name="ABLATION_PREFLIGHT_TEST", kind="interpolate", count=48, seed=18001, global_seen=seen)

    torch.manual_seed(1234)
    dense = arch.PrunedBrainSpikeNet()
    dense_initial = {k: v.detach().clone() for k, v in dense.state_dict().items() if k.startswith("W_")}
    torch.manual_seed(1234)
    pruned = arch.PrunedBrainSpikeNet()
    for name, tensor in dense_initial.items():
        if not torch.equal(tensor, pruned.state_dict()[name]):
            raise AssertionError(f"SNN initial weights differ at {name}")

    dense, _ = train_variant(dense, train, test, epochs=2, batch_size=32, seed=1234, label="DENSE_RSNN", enable_pruning=False)
    if abs(dense.sparsity_report()["overall"]) > 1e-12:
        raise AssertionError("dense SNN unexpectedly sparse")

    opt = torch.optim.AdamW(pruned.parameters(), lr=arch.LR, weight_decay=arch.WEIGHT_DECAY)
    report = pruned.prune_synapses(0.30, opt)
    if abs(report["overall"] - 0.30) > 0.001:
        raise AssertionError(report)

    print(
        "DENSE_PRUNED_PREFLIGHT_PASS",
        json.dumps({"dense_sparsity": dense.sparsity_report(), "pruned_sparsity": report}, sort_keys=True),
        flush=True,
    )


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
    parser.add_argument("--output-dir", default="deepmind-dense-vs-pruned-output")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    dm_algebra = dm.prepare_deepmind()
    if args.preflight:
        preflight(dm_algebra)
        return

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) < 3:
        raise ValueError("ablation requires at least 3 seeds")
    if args.epochs < arch.PRUNE_END:
        raise ValueError("epochs must cover the full pruning schedule")

    seen: set = set()
    train = dm.build_bank(dm_algebra, name="TRAIN_DEEPMIND", kind="train", count=args.train_size, seed=7301, global_seen=seen)
    interp = dm.build_bank(dm_algebra, name="OFFICIAL_INTERPOLATE", kind="interpolate", count=args.eval_size, seed=8301, global_seen=seen)
    stress10 = dm.build_bank(dm_algebra, name="STRESS_ENTROPY_10", kind="entropy10", count=args.eval_size, seed=9301, global_seen=seen)
    stress12 = dm.build_bank(dm_algebra, name="STRESS_ENTROPY_12", kind="entropy12", count=args.eval_size, seed=10301, global_seen=seen)
    ill_pool = dm.build_bank(dm_algebra, name="ILL_POOL_ENTROPY_12", kind="entropy12", count=args.ill_pool_size, seed=11301, global_seen=seen)
    ill_ids = torch.argsort(ill_pool.condition, descending=True)[: args.ill_size]
    ill = dm.subset_bank(ill_pool, ill_ids, "ILL_CONDITIONED")
    banks = {interp.name: interp, stress10.name: stress10, stress12.name: stress12, ill.name: ill}

    p_snn = arch.count_parameters(arch.PrunedBrainSpikeNet())
    p_mlp = arch.count_parameters(arch.OrdinaryMLP())
    if p_snn != 26880 or p_mlp != p_snn:
        raise AssertionError(f"PARAMETER_MATCH_FAILED SNN={p_snn} MLP={p_mlp}")
    print(f"PARAMETER_MATCH DENSE_RSNN={p_snn} PRUNED_RSNN={p_snn} MLP={p_mlp}", flush=True)
    print("ABLATION_CONTRACT dense/pruned SNN share exact architecture, optimizer, data, batches, seeds and initial weights; only pruning differs", flush=True)
    print("DATA_CONTRACT official google-deepmind/mathematics_dataset linear_2d only; project synthetic data=0", flush=True)

    all_runs: List[Dict[str, object]] = []
    for seed in seeds:
        variants = (
            ("DENSE_RSNN", arch.PrunedBrainSpikeNet, False),
            ("PRUNED_RSNN", arch.PrunedBrainSpikeNet, True),
            ("ORDINARY_MLP", arch.OrdinaryMLP, False),
        )
        for label, cls, enable_pruning in variants:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = cls()
            model, result = train_variant(
                model,
                train,
                interp,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=seed,
                label=label,
                enable_pruning=enable_pruning,
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
        "project_synthetic_examples": 0,
        "parameter_match": {"DENSE_RSNN": p_snn, "PRUNED_RSNN": p_snn, "ORDINARY_MLP": p_mlp},
        "effective_active_weights_after_training": {
            "DENSE_RSNN": p_snn,
            "PRUNED_RSNN": int(round(p_snn * (1.0 - arch.FINAL_SPARSITY))),
            "ORDINARY_MLP": p_mlp,
        },
        "banks": {
            train.name: dm.bank_manifest(train),
            **{name: dm.bank_manifest(bank) for name, bank in banks.items()},
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"manifest": manifest, "runs": all_runs, "aggregate": summary}, indent=2, sort_keys=True))
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    lines = [
        "DeepMind Dense RSNN vs Pruned RSNN vs Ordinary MLP ablation",
        f"DeepMind commit={dm.DM_COMMIT}",
        f"Seeds={seeds}",
        f"Train={len(train.x)} OfficialInterp={len(interp.x)} Stress10={len(stress10.x)} Stress12={len(stress12.x)} IllConditioned={len(ill.x)}",
        f"Params each={p_snn}; pruned active≈{int(round(p_snn * 0.70))}",
        "Project synthetic examples=0",
    ]
    for bank in bank_names:
        lines.append(f"{bank} wins_by_MAE={summary['seed_wins_by_mae'][bank]}")
        for label in LABELS:
            a = summary[label][bank]
            lines.append(
                f"  {label}: strict0.1={a['strict_0_1']['mean']:.6f} strict1={a['strict_1_0']['mean']:.6f} "
                f"within5={a['within_5_0']['mean']:.6f} mae={a['mae']['mean']:.6f} rmse={a['rmse']['mean']:.6f} "
                f"p95={a['p95_sample_max_error']['mean']:.6f} residual={a['equation_residual_mean']['mean']:.6f}"
            )
        dvp = summary["dense_vs_pruned"][bank]
        lines.append(
            f"  PRUNING_DELTA: MAE={dvp['pruned_mae_change_pct_vs_dense']:+.2f}% vs dense; "
            f"strict1={dvp['strict_1_point_change_pruned_minus_dense']:+.2f} percentage-points"
        )
    for label in LABELS:
        lines.append(f"TRAIN_SECONDS {label}: {summary[label]['train_seconds']['mean']:.3f}±{summary[label]['train_seconds']['std']:.3f}")

    (out_dir / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    print("FINAL_ABLATION", json.dumps(summary, sort_keys=True), flush=True)
    print("DENSE_VS_PRUNED_DEEPMIND_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
