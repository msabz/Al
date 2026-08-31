#!/usr/bin/env python3
"""Two controlled DeepMind distillation experiments for the RSNN.

Experiment A: DENSE_RSNN -> PRUNED_RSNN_30_DISTILLED
  Controls: the dense teacher and a plain supervised PRUNED_RSNN_30 trained
  from the exact same RSNN initialization. This isolates the effect of
  distillation from the effect of 30% pruning itself.

Experiment B: ORDINARY_MLP -> DENSE_RSNN_DISTILLED -> PRUNED_RSNN_30_STAGE2
  A true two-stage cross-architecture knowledge-transfer chain.

Fairness contract
-----------------
- Same pinned Google DeepMind mathematics_dataset source and linear_2d banks
  used by the existing harsh benchmark.
- Same train/eval sizes, seeds, epochs, batches, optimizer, scheduler, target
  scaling, RSNN architecture, pruning schedule, masks, and evaluation metrics.
- Distillation is the only added training signal: 50% ground-truth SmoothL1
  + 50% teacher-output SmoothL1, matching the Colab experiments.
- No project synthetic examples are used.
- 30% pruning uses the repository's persistent binary masks, zeros gradients
  and optimizer state for pruned weights, and reapplies masks after each step.
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
FINAL_SPARSITY = 0.30
HARD_WEIGHT = 0.50
SOFT_WEIGHT = 0.50
METRICS = (
    "mae", "rmse", "median_sample_max_error", "p95_sample_max_error",
    "max_abs_error", "strict_0_1", "strict_0_5", "strict_1_0",
    "within_5_0", "relative_mae", "equation_residual_mean",
    "equation_residual_p95",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def assert_same_rsnn_initialization(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], context: str) -> None:
    for name in ("W_in", "W_rec", "W_out", "M_in", "M_rec", "M_out"):
        if not torch.equal(a[name], b[name]):
            raise AssertionError(f"RSNN initialization mismatch {context} param={name}")


def maybe_prune(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    final_sparsity: float,
    seed: int,
    label: str,
) -> None:
    if not isinstance(model, arch.PrunedBrainSpikeNet):
        return
    target = pruning_target(epoch, final_sparsity)
    if target is None:
        return
    report = model.prune_synapses(target, optimizer)
    print(
        f"PRUNE seed={seed} label={label} epoch={epoch} target={target:.4f} "
        f"actual={json.dumps(report, sort_keys=True)}",
        flush=True,
    )


def finalize_sparsity(model: nn.Module, expected: float, seed: int, label: str) -> Dict[str, float] | None:
    if not isinstance(model, arch.PrunedBrainSpikeNet):
        return None
    report = model.sparsity_report()
    for key in ("W_in", "W_rec", "W_out", "overall"):
        if abs(float(report[key]) - expected) > 0.002:
            raise AssertionError(
                f"final sparsity mismatch seed={seed} label={label} {key}={report[key]} expected={expected}"
            )
    return report


def train_supervised(
    model: nn.Module,
    train: dm.Bank,
    monitor: dm.Bank,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    label: str,
    final_sparsity: float = 0.0,
) -> Tuple[nn.Module, Dict[str, object]]:
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arch.LR, weight_decay=arch.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=arch.ETA_MIN)
    criterion = nn.SmoothL1Loss(beta=1.0)
    checkpoints = {1, 50, 100, 150, 200, 250, epochs}
    checkpoints = {e for e in checkpoints if 1 <= e <= epochs}
    history: List[Dict[str, object]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        maybe_prune(
            model, optimizer, epoch=epoch, final_sparsity=final_sparsity, seed=seed, label=label
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
                raise RuntimeError(f"non-finite supervised loss seed={seed} label={label} epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if isinstance(model, arch.PrunedBrainSpikeNet):
                model.zero_pruned_gradients_()
            optimizer.step()
            if isinstance(model, arch.PrunedBrainSpikeNet):
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
            }
            if isinstance(model, arch.PrunedBrainSpikeNet):
                row["sparsity"] = model.sparsity_report()
            history.append(row)
            print(f"CHECKPOINT seed={seed} label={label} {json.dumps(row, sort_keys=True)}", flush=True)

    elapsed = time.perf_counter() - started
    sparsity = finalize_sparsity(model, final_sparsity, seed, label)
    result: Dict[str, object] = {
        "seed": seed,
        "label": label,
        "training": "supervised",
        "params": arch.count_parameters(model),
        "active_weights": int(round(arch.count_parameters(model) * (1.0 - final_sparsity))) if isinstance(model, arch.PrunedBrainSpikeNet) else arch.count_parameters(model),
        "train_seconds": elapsed,
        "history": history,
    }
    if sparsity is not None:
        result["sparsity"] = sparsity
    return model, result


def train_distilled(
    student: arch.PrunedBrainSpikeNet,
    teacher: nn.Module,
    train: dm.Bank,
    monitor: dm.Bank,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    label: str,
    teacher_label: str,
    final_sparsity: float,
) -> Tuple[arch.PrunedBrainSpikeNet, Dict[str, object]]:
    student.to(DEVICE)
    teacher.to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(student.parameters(), lr=arch.LR, weight_decay=arch.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=arch.ETA_MIN)
    criterion = nn.SmoothL1Loss(beta=1.0)
    checkpoints = {1, 50, 100, 150, 200, 250, epochs}
    checkpoints = {e for e in checkpoints if 1 <= e <= epochs}
    history: List[Dict[str, object]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        maybe_prune(
            student, optimizer, epoch=epoch, final_sparsity=final_sparsity, seed=seed, label=label
        )
        student.train()
        hard_running = 0.0
        soft_running = 0.0
        total_running = 0.0
        seen = 0
        for ids in arch.deterministic_batches(len(train.x), batch_size, seed, epoch):
            bx = train.x[ids].to(DEVICE)
            by = train.y_scaled[ids].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_pred = teacher(bx)
            student_pred = student(bx)
            hard_loss = criterion(student_pred, by)
            soft_loss = criterion(student_pred, teacher_pred)
            total_loss = HARD_WEIGHT * hard_loss + SOFT_WEIGHT * soft_loss
            if not torch.isfinite(total_loss):
                raise RuntimeError(f"non-finite distillation loss seed={seed} label={label} epoch={epoch}")
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            student.zero_pruned_gradients_()
            optimizer.step()
            student.apply_masks_()
            n = len(ids)
            hard_running += float(hard_loss.detach()) * n
            soft_running += float(soft_loss.detach()) * n
            total_running += float(total_loss.detach()) * n
            seen += n
        scheduler.step()

        if epoch in checkpoints:
            row: Dict[str, object] = {
                "epoch": epoch,
                "total_loss": total_running / max(seen, 1),
                "hard_loss": hard_running / max(seen, 1),
                "soft_loss": soft_running / max(seen, 1),
                "lr": optimizer.param_groups[0]["lr"],
                "teacher": teacher_label,
                "official_interpolate": dm.evaluate_raw(student, monitor),
                "sparsity": student.sparsity_report(),
            }
            history.append(row)
            print(f"CHECKPOINT seed={seed} label={label} {json.dumps(row, sort_keys=True)}", flush=True)

    elapsed = time.perf_counter() - started
    sparsity = finalize_sparsity(student, final_sparsity, seed, label)
    return student, {
        "seed": seed,
        "label": label,
        "training": "distillation",
        "teacher": teacher_label,
        "hard_weight": HARD_WEIGHT,
        "soft_weight": SOFT_WEIGHT,
        "params": arch.count_parameters(student),
        "active_weights": int(round(arch.count_parameters(student) * (1.0 - final_sparsity))),
        "train_seconds": elapsed,
        "sparsity": sparsity,
        "history": history,
    }


def build_banks(args) -> Tuple[dm.Bank, Dict[str, dm.Bank], Dict[str, object]]:
    dm_algebra = dm.prepare_deepmind()
    seen: set = set()
    train = dm.build_bank(dm_algebra, name="TRAIN_DEEPMIND", kind="train", count=args.train_size, seed=7301, global_seen=seen)
    interp = dm.build_bank(dm_algebra, name="OFFICIAL_INTERPOLATE", kind="interpolate", count=args.eval_size, seed=8301, global_seen=seen)
    stress10 = dm.build_bank(dm_algebra, name="STRESS_ENTROPY_10", kind="entropy10", count=args.eval_size, seed=9301, global_seen=seen)
    stress12 = dm.build_bank(dm_algebra, name="STRESS_ENTROPY_12", kind="entropy12", count=args.eval_size, seed=10301, global_seen=seen)
    ill_pool = dm.build_bank(dm_algebra, name="ILL_POOL_ENTROPY_12", kind="entropy12", count=args.ill_pool_size, seed=11301, global_seen=seen)
    ill_ids = torch.argsort(ill_pool.condition, descending=True)[: args.ill_size]
    ill = dm.subset_bank(ill_pool, ill_ids, "ILL_CONDITIONED")
    banks = {interp.name: interp, stress10.name: stress10, stress12.name: stress12, ill.name: ill}
    manifest = {
        "deepmind_repo": dm.DM_REPO,
        "deepmind_commit": dm.DM_COMMIT,
        "project_synthetic_examples": 0,
        "banks": {
            train.name: dm.bank_manifest(train),
            **{name: dm.bank_manifest(bank) for name, bank in banks.items()},
        },
    }
    return train, banks, manifest


def attach_evaluations(result: Dict[str, object], model: nn.Module, banks: Dict[str, dm.Bank]) -> None:
    result["evaluations"] = {name: dm.evaluate_raw(model, bank) for name, bank in banks.items()}
    print(
        f"SEED_FINAL seed={result['seed']} label={result['label']} "
        f"{json.dumps(result['evaluations'], sort_keys=True)}",
        flush=True,
    )


def aggregate_by_label(runs: List[Dict[str, object]], labels: Sequence[str], bank_names: Sequence[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    seeds = sorted({int(r["seed"]) for r in runs})
    for label in labels:
        rows = [r for r in runs if r["label"] == label]
        if len(rows) != len(seeds):
            raise AssertionError(f"missing runs label={label}: {len(rows)} != {len(seeds)}")
        entry: Dict[str, object] = {
            "params": int(rows[0]["params"]),
            "active_weights": int(rows[0]["active_weights"]),
            "train_seconds": {
                "mean": mean(float(r["train_seconds"]) for r in rows),
                "std": pstdev(float(r["train_seconds"]) for r in rows) if len(rows) > 1 else 0.0,
            },
            "banks": {},
        }
        for bank in bank_names:
            metrics: Dict[str, object] = {}
            for metric in METRICS:
                vals = [float(r["evaluations"][bank][metric]) for r in rows]
                metrics[metric] = {
                    "mean": mean(vals),
                    "std": pstdev(vals) if len(vals) > 1 else 0.0,
                    "min": min(vals),
                    "max": max(vals),
                }
            entry["banks"][bank] = metrics
        out[label] = entry
    return out


def paired_compare(runs: List[Dict[str, object]], reference: str, candidate: str, bank_names: Sequence[str]) -> Dict[str, object]:
    refs = {int(r["seed"]): r for r in runs if r["label"] == reference}
    cands = {int(r["seed"]): r for r in runs if r["label"] == candidate}
    if set(refs) != set(cands):
        raise AssertionError(f"paired seed mismatch {reference} vs {candidate}")
    out: Dict[str, object] = {}
    for bank in bank_names:
        mae_pct: List[float] = []
        strict1_pp: List[float] = []
        within5_pp: List[float] = []
        residual_pct: List[float] = []
        wins = 0
        for seed in sorted(refs):
            a = refs[seed]["evaluations"][bank]
            b = cands[seed]["evaluations"][bank]
            a_mae = float(a["mae"])
            b_mae = float(b["mae"])
            mae_pct.append(100.0 * (b_mae / a_mae - 1.0))
            strict1_pp.append(100.0 * (float(b["strict_1_0"]) - float(a["strict_1_0"])))
            within5_pp.append(100.0 * (float(b["within_5_0"]) - float(a["within_5_0"])))
            a_res = float(a["equation_residual_mean"])
            b_res = float(b["equation_residual_mean"])
            residual_pct.append(100.0 * (b_res / a_res - 1.0))
            if b_mae < a_mae:
                wins += 1
        out[bank] = {
            "reference": reference,
            "candidate": candidate,
            "mae_change_pct_mean": mean(mae_pct),
            "mae_change_pct_std": pstdev(mae_pct),
            "strict1_change_pp_mean": mean(strict1_pp),
            "strict1_change_pp_std": pstdev(strict1_pp),
            "within5_change_pp_mean": mean(within5_pp),
            "residual_change_pct_mean": mean(residual_pct),
            "mae_wins": wins,
            "mae_losses": len(refs) - wins,
        }
    return out


def run_dense_to_pruned(args, train: dm.Bank, banks: Dict[str, dm.Bank]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    labels = ["DENSE_RSNN_TEACHER", "P30_SUPERVISED_CONTROL", "P30_DISTILLED_FROM_DENSE"]
    all_runs: List[Dict[str, object]] = []
    interp = banks["OFFICIAL_INTERPOLATE"]

    for seed in args.seed_values:
        set_seed(seed)
        base = arch.PrunedBrainSpikeNet()
        base_state = clone_state(base)

        dense = arch.PrunedBrainSpikeNet()
        dense.load_state_dict(base_state)
        assert_same_rsnn_initialization(base_state, clone_state(dense), f"seed={seed} dense")
        dense, dense_result = train_supervised(
            dense, train, interp, epochs=args.epochs, batch_size=args.batch_size,
            seed=seed, label="DENSE_RSNN_TEACHER", final_sparsity=0.0,
        )
        attach_evaluations(dense_result, dense, banks)
        all_runs.append(dense_result)

        plain = arch.PrunedBrainSpikeNet()
        plain.load_state_dict(base_state)
        assert_same_rsnn_initialization(base_state, clone_state(plain), f"seed={seed} plain-p30")
        plain, plain_result = train_supervised(
            plain, train, interp, epochs=args.epochs, batch_size=args.batch_size,
            seed=seed, label="P30_SUPERVISED_CONTROL", final_sparsity=FINAL_SPARSITY,
        )
        attach_evaluations(plain_result, plain, banks)
        all_runs.append(plain_result)
        del plain

        student = arch.PrunedBrainSpikeNet()
        student.load_state_dict(base_state)
        assert_same_rsnn_initialization(base_state, clone_state(student), f"seed={seed} distilled-p30")
        student, student_result = train_distilled(
            student, dense, train, interp, epochs=args.epochs, batch_size=args.batch_size,
            seed=seed, label="P30_DISTILLED_FROM_DENSE", teacher_label="DENSE_RSNN_TEACHER",
            final_sparsity=FINAL_SPARSITY,
        )
        attach_evaluations(student_result, student, banks)
        all_runs.append(student_result)
        del dense, student, base

    aggregate = aggregate_by_label(all_runs, labels, list(banks))
    comparisons = {
        "plain_p30_vs_dense": paired_compare(all_runs, "DENSE_RSNN_TEACHER", "P30_SUPERVISED_CONTROL", list(banks)),
        "distilled_p30_vs_dense": paired_compare(all_runs, "DENSE_RSNN_TEACHER", "P30_DISTILLED_FROM_DENSE", list(banks)),
        "distilled_p30_vs_plain_p30": paired_compare(all_runs, "P30_SUPERVISED_CONTROL", "P30_DISTILLED_FROM_DENSE", list(banks)),
    }
    return all_runs, {"models": aggregate, "paired": comparisons}


def run_mlp_chain(args, train: dm.Bank, banks: Dict[str, dm.Bank]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    labels = ["MLP_PRIMARY_TEACHER", "DENSE_RSNN_DISTILLED_FROM_MLP", "P30_STAGE2_DISTILLED_FROM_RSNN"]
    all_runs: List[Dict[str, object]] = []
    interp = banks["OFFICIAL_INTERPOLATE"]

    for seed in args.seed_values:
        set_seed(seed)
        mlp = arch.OrdinaryMLP()
        mlp, mlp_result = train_supervised(
            mlp, train, interp, epochs=args.epochs, batch_size=args.batch_size,
            seed=seed, label="MLP_PRIMARY_TEACHER", final_sparsity=0.0,
        )
        attach_evaluations(mlp_result, mlp, banks)
        all_runs.append(mlp_result)

        set_seed(seed)
        rsnn_base = arch.PrunedBrainSpikeNet()
        rsnn_state = clone_state(rsnn_base)

        dense_student = arch.PrunedBrainSpikeNet()
        dense_student.load_state_dict(rsnn_state)
        assert_same_rsnn_initialization(rsnn_state, clone_state(dense_student), f"seed={seed} stage1-dense")
        dense_student, dense_result = train_distilled(
            dense_student, mlp, train, interp, epochs=args.epochs, batch_size=args.batch_size,
            seed=seed, label="DENSE_RSNN_DISTILLED_FROM_MLP", teacher_label="MLP_PRIMARY_TEACHER",
            final_sparsity=0.0,
        )
        attach_evaluations(dense_result, dense_student, banks)
        all_runs.append(dense_result)

        p30_student = arch.PrunedBrainSpikeNet()
        p30_student.load_state_dict(rsnn_state)
        assert_same_rsnn_initialization(rsnn_state, clone_state(p30_student), f"seed={seed} stage2-p30")
        p30_student, p30_result = train_distilled(
            p30_student, dense_student, train, interp, epochs=args.epochs, batch_size=args.batch_size,
            seed=seed, label="P30_STAGE2_DISTILLED_FROM_RSNN", teacher_label="DENSE_RSNN_DISTILLED_FROM_MLP",
            final_sparsity=FINAL_SPARSITY,
        )
        attach_evaluations(p30_result, p30_student, banks)
        all_runs.append(p30_result)
        del mlp, dense_student, p30_student, rsnn_base

    aggregate = aggregate_by_label(all_runs, labels, list(banks))
    comparisons = {
        "dense_stage1_vs_mlp": paired_compare(all_runs, "MLP_PRIMARY_TEACHER", "DENSE_RSNN_DISTILLED_FROM_MLP", list(banks)),
        "p30_stage2_vs_dense_stage1": paired_compare(all_runs, "DENSE_RSNN_DISTILLED_FROM_MLP", "P30_STAGE2_DISTILLED_FROM_RSNN", list(banks)),
        "p30_stage2_vs_mlp": paired_compare(all_runs, "MLP_PRIMARY_TEACHER", "P30_STAGE2_DISTILLED_FROM_RSNN", list(banks)),
    }
    return all_runs, {"models": aggregate, "paired": comparisons}


def write_outputs(args, manifest: Dict[str, object], runs: List[Dict[str, object]], summary: Dict[str, object]) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **manifest,
        "experiment": args.experiment,
        "seeds": args.seed_values,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hard_weight": HARD_WEIGHT,
        "soft_weight": SOFT_WEIGHT,
        "final_sparsity": FINAL_SPARSITY,
        "rsnn_params": arch.count_parameters(arch.PrunedBrainSpikeNet()),
        "mlp_params": arch.count_parameters(arch.OrdinaryMLP()),
        "p30_active_weights": int(round(arch.count_parameters(arch.PrunedBrainSpikeNet()) * (1.0 - FINAL_SPARSITY))),
    }
    payload = {"manifest": manifest, "runs": runs, "aggregate": summary}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    official = "OFFICIAL_INTERPOLATE"
    lines = [
        f"DeepMind RSNN distillation experiment: {args.experiment}",
        f"DeepMind commit={dm.DM_COMMIT}",
        f"Seeds={args.seed_values}",
        f"Epochs={args.epochs} batch={args.batch_size}",
        f"Distillation loss={HARD_WEIGHT:.2f}*ground_truth + {SOFT_WEIGHT:.2f}*teacher",
        f"RSNN params={manifest['rsnn_params']} P30 active={manifest['p30_active_weights']}",
        "Project synthetic examples=0",
        "",
        "OFFICIAL_INTERPOLATE:",
    ]
    for label, entry in summary["models"].items():
        m = entry["banks"][official]
        lines.append(
            f"  {label}: active={entry['active_weights']} MAE={m['mae']['mean']:.6f}±{m['mae']['std']:.6f} "
            f"RMSE={m['rmse']['mean']:.6f} strict0.1={100*m['strict_0_1']['mean']:.3f}% "
            f"strict1={100*m['strict_1_0']['mean']:.3f}% within5={100*m['within_5_0']['mean']:.3f}% "
            f"residual={m['equation_residual_mean']['mean']:.6f}"
        )
    lines.append("")
    lines.append("PAIRED COMPARISONS (candidate relative to reference):")
    for comparison_name, banks in summary["paired"].items():
        p = banks[official]
        lines.append(
            f"  {comparison_name}: MAE={p['mae_change_pct_mean']:+.3f}%±{p['mae_change_pct_std']:.3f}% "
            f"strict1={p['strict1_change_pp_mean']:+.3f}pp within5={p['within5_change_pp_mean']:+.3f}pp "
            f"residual={p['residual_change_pct_mean']:+.3f}% MAE_wins={p['mae_wins']}/{len(args.seed_values)}"
        )
    lines.append("")
    for bank in summary["models"][next(iter(summary["models"]))]["banks"]:
        lines.append(bank)
        for label, entry in summary["models"].items():
            m = entry["banks"][bank]
            lines.append(
                f"  {label}: MAE={m['mae']['mean']:.6f} strict1={100*m['strict_1_0']['mean']:.3f}% "
                f"within5={100*m['within_5_0']['mean']:.3f}% residual={m['equation_residual_mean']['mean']:.6f}"
            )
    (out_dir / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    print("FINAL_DISTILLATION_RESULT", json.dumps(summary, sort_keys=True), flush=True)
    print(f"DISTILLATION_EXPERIMENT_COMPLETE experiment={args.experiment}", flush=True)


def preflight(experiment: str) -> None:
    params_rsnn = arch.count_parameters(arch.PrunedBrainSpikeNet())
    params_mlp = arch.count_parameters(arch.OrdinaryMLP())
    if params_rsnn != 26880 or params_mlp != 26880:
        raise AssertionError(f"parameter contract failed rsnn={params_rsnn} mlp={params_mlp}")
    set_seed(1234)
    a = arch.PrunedBrainSpikeNet()
    state = clone_state(a)
    b = arch.PrunedBrainSpikeNet()
    b.load_state_dict(state)
    assert_same_rsnn_initialization(state, clone_state(b), "preflight")
    opt = torch.optim.AdamW(b.parameters(), lr=arch.LR, weight_decay=arch.WEIGHT_DECAY)
    report = b.prune_synapses(FINAL_SPARSITY, opt)
    if abs(float(report["overall"]) - FINAL_SPARSITY) > 0.002:
        raise AssertionError(f"P30 mask contract failed {report}")
    print(
        f"DISTILLATION_PREFLIGHT_PASS experiment={experiment} params=26880 p30_active=18816 "
        f"hard_weight={HARD_WEIGHT} soft_weight={SOFT_WEIGHT} persistent_masks=true",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=("dense-to-pruned", "mlp-chain"))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--eval-size", type=int, default=2048)
    parser.add_argument("--ill-pool-size", type=int, default=4096)
    parser.add_argument("--ill-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    args.seed_values = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    if args.preflight:
        preflight(args.experiment)
        return
    if len(args.seed_values) < 3:
        raise ValueError("distillation test requires at least 3 seeds")
    if args.epochs < arch.PRUNE_END:
        raise ValueError("epochs must cover the full 30% pruning schedule")

    train, banks, manifest = build_banks(args)
    print(
        f"DISTILLATION_CONTRACT experiment={args.experiment} deepmind_commit={dm.DM_COMMIT} "
        f"train={len(train.x)} eval={args.eval_size} seeds={args.seed_values} epochs={args.epochs} "
        f"hard={HARD_WEIGHT} soft={SOFT_WEIGHT} final_sparsity={FINAL_SPARSITY}",
        flush=True,
    )
    print("DATA_CONTRACT official google-deepmind/mathematics_dataset linear_2d only; project synthetic data=0", flush=True)

    if args.experiment == "dense-to-pruned":
        runs, summary = run_dense_to_pruned(args, train, banks)
    else:
        runs, summary = run_mlp_chain(args, train, banks)
    write_outputs(args, manifest, runs, summary)


if __name__ == "__main__":
    main()
