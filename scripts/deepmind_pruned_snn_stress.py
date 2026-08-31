#!/usr/bin/env python3
"""Harsh multi-seed DeepMind benchmark for the pruned RSNN vs ordinary MLP.

The model task remains exactly six continuous coefficients -> two continuous
solutions for a 2x2 linear system. Data comes only from the official
Google DeepMind mathematics_dataset generator. No project synthetic equations
are used.

Banks:
- TRAIN: official algebra.train(...)["linear_2d"] generator.
- OFFICIAL_INTERPOLATE: official algebra.test()["linear_2d"] split.
- STRESS_ENTROPY_10: same official DeepMind generator at fixed entropy 10.
- STRESS_ENTROPY_12: same official DeepMind generator at fixed entropy 12.
- ILL_CONDITIONED: highest-condition-number examples from an independent
  entropy-12 DeepMind pool.

Note: mathematics_dataset does not expose a named linear_2d test_extra split;
therefore the entropy-10/12 banks are explicitly reported as stress banks, not
as official named extrapolation splits.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import sympy as sp
import torch
import torch.nn as nn

import pruned_snn_vs_mlp as arch

DM_REPO = "https://github.com/google-deepmind/mathematics_dataset.git"
DM_COMMIT = "427f45075f84b8b9774950196ad63867ca20ffb3"
DM_ROOT = Path("/tmp/deepmind-mathematics-dataset-snn")
TARGET_SCALE = 100.0
MAX_ABS_TARGET = 300.0
DEVICE = torch.device("cpu")


@dataclass
class Bank:
    name: str
    x: torch.Tensor
    y_raw: torch.Tensor
    condition: torch.Tensor
    sample_questions: List[str]
    attempts: int

    @property
    def y_scaled(self) -> torch.Tensor:
        return self.y_raw / TARGET_SCALE


def prepare_deepmind():
    if DM_ROOT.exists():
        shutil.rmtree(DM_ROOT)
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", DM_REPO, str(DM_ROOT)],
        check=True,
    )
    head = subprocess.check_output(
        ["git", "-C", str(DM_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != DM_COMMIT:
        raise RuntimeError(f"DeepMind source drift: {head} != {DM_COMMIT}")
    sys.path.insert(0, str(DM_ROOT))
    from mathematics_dataset.modules import algebra as dm_algebra
    print(f"DEEPMIND_SOURCE_OK commit={head}", flush=True)
    return dm_algebra


def _canonical_row(a: float, b: float, c: float) -> np.ndarray:
    row = np.asarray([a, b, c], dtype=np.float64)
    scale = float(np.max(np.abs(row)))
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError("zero equation row")
    row = row / scale
    for value in row[:2]:
        if abs(value) > 1e-12:
            if value < 0:
                row = -row
            break
    return row


def parse_deepmind_linear_2d(problem) -> Tuple[np.ndarray, np.ndarray, float]:
    question = str(problem.question).strip()
    match = re.match(r"^Solve\s+(.+)\s+for\s+([A-Za-z]+)\.$", question)
    if not match:
        raise ValueError("unsupported question template")
    body, asked = match.group(1), match.group(2)
    parts = [p.strip() for p in re.split(r"\s*(?:,|\band\b)\s*", body) if "=" in p]
    if len(parts) != 2:
        raise ValueError(f"expected two equations: {question}")

    equations = []
    symbols = set()
    for text in parts:
        lhs_text, rhs_text = text.split("=", 1)
        lhs = sp.sympify(lhs_text.strip().replace("^", "**"))
        rhs = sp.sympify(rhs_text.strip().replace("^", "**"))
        expr = sp.expand(lhs - rhs)
        equations.append(expr)
        symbols.update(expr.free_symbols)

    syms = sorted(symbols, key=lambda s: str(s))
    if len(syms) != 2:
        raise ValueError(f"symbol count={len(syms)}")

    rows = []
    raw_rows = []
    for expr in equations:
        poly = sp.Poly(expr, *syms)
        if poly.total_degree() > 1:
            raise ValueError("nonlinear expression")
        a = float(sp.N(poly.coeff_monomial(syms[0]), 18))
        b = float(sp.N(poly.coeff_monomial(syms[1]), 18))
        constant = float(sp.N(poly.coeff_monomial(1), 18))
        c = -constant
        raw_rows.append(np.asarray([a, b, c], dtype=np.float64))
        rows.append(_canonical_row(a, b, c))

    # Canonicalize equation order. This removes irrelevant textual equation
    # permutation while preserving the exact mathematical system.
    rows = sorted(rows, key=lambda r: tuple(np.round(r, 14).tolist()))
    r1, r2 = rows
    a, b, c = r1.tolist()
    d, e, f = r2.tolist()
    det = a * e - b * d
    if abs(det) <= 1e-10:
        raise ValueError("singular or numerically degenerate system")
    sol0 = (c * e - b * f) / det
    sol1 = (a * f - c * d) / det
    y = np.asarray([sol0, sol1], dtype=np.float64)
    if not np.all(np.isfinite(y)) or float(np.max(np.abs(y))) > MAX_ABS_TARGET:
        raise ValueError("target outside declared support")

    # Verify the official answer against the solution of the parsed system.
    asked_idx = next((i for i, s in enumerate(syms) if str(s) == asked), None)
    if asked_idx is None:
        raise ValueError("asked variable missing")
    official_answer = float(sp.N(sp.sympify(str(problem.answer)), 18))
    if not math.isfinite(official_answer) or abs(official_answer - y[asked_idx]) > 1e-5:
        # Canonical equation order does not change variable order, so this is a
        # real parser/data consistency failure.
        raise ValueError("official answer verification failed")

    x = np.asarray([a, b, c, d, e, f], dtype=np.float32)
    cond = float(np.linalg.cond(np.asarray([[a, b], [d, e]], dtype=np.float64)))
    if not math.isfinite(cond):
        raise ValueError("non-finite condition number")
    return x, y.astype(np.float32), cond


def generator_for(dm_algebra, kind: str):
    if kind == "train":
        return dm_algebra.train(lambda entropy_range: entropy_range)["linear_2d"]
    if kind == "interpolate":
        return dm_algebra.test()["linear_2d"]
    if kind == "entropy10":
        return dm_algebra._make_modules((10, 10))["linear_2d"]
    if kind == "entropy12":
        return dm_algebra._make_modules((12, 12))["linear_2d"]
    raise ValueError(kind)


def build_bank(
    dm_algebra,
    *,
    name: str,
    kind: str,
    count: int,
    seed: int,
    global_seen: set,
) -> Bank:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gen = generator_for(dm_algebra, kind)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    conds: List[float] = []
    questions: List[str] = []
    attempts = 0
    max_attempts = max(5000, count * 500)
    while len(xs) < count and attempts < max_attempts:
        attempts += 1
        try:
            problem = gen()
            x, y, cond = parse_deepmind_linear_2d(problem)
            key = tuple(np.round(x.astype(np.float64), 10)) + tuple(np.round(y.astype(np.float64), 8))
            if key in global_seen:
                continue
            global_seen.add(key)
            xs.append(x)
            ys.append(y)
            conds.append(cond)
            if len(questions) < 8:
                questions.append(str(problem.question))
            if len(xs) % 1024 == 0 or len(xs) == count:
                print(f"DATA_BANK {name} accepted={len(xs)}/{count} attempts={attempts}", flush=True)
        except Exception:
            continue
    if len(xs) != count:
        raise RuntimeError(f"Could not build {name}: {len(xs)}/{count} after {attempts} attempts")
    return Bank(
        name=name,
        x=torch.tensor(np.stack(xs), dtype=torch.float32),
        y_raw=torch.tensor(np.stack(ys), dtype=torch.float32),
        condition=torch.tensor(np.asarray(conds), dtype=torch.float32),
        sample_questions=questions,
        attempts=attempts,
    )


def subset_bank(bank: Bank, ids: torch.Tensor, name: str) -> Bank:
    return Bank(
        name=name,
        x=bank.x[ids].clone(),
        y_raw=bank.y_raw[ids].clone(),
        condition=bank.condition[ids].clone(),
        sample_questions=list(bank.sample_questions),
        attempts=bank.attempts,
    )


@torch.no_grad()
def evaluate_raw(model: nn.Module, bank: Bank, batch_size: int = 512) -> Dict[str, float]:
    model.eval()
    out = []
    for start in range(0, len(bank.x), batch_size):
        out.append(model(bank.x[start:start + batch_size].to(DEVICE)).cpu())
    pred_raw = torch.cat(out, dim=0) * TARGET_SCALE
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


def train_one(
    model: nn.Module,
    train: Bank,
    monitor: Bank,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    label: str,
) -> Tuple[nn.Module, Dict[str, object]]:
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=arch.LR, weight_decay=arch.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=arch.ETA_MIN
    )
    criterion = nn.SmoothL1Loss(beta=1.0)
    checkpoints = {1, 25, 50, 100, 150, 200, 250, epochs}
    checkpoints = {e for e in checkpoints if 1 <= e <= epochs}
    history = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if isinstance(model, arch.PrunedBrainSpikeNet):
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
                model.zero_pruned_gradients_()
            optimizer.step()
            if isinstance(model, arch.PrunedBrainSpikeNet):
                model.apply_masks_()
            running += float(loss.detach()) * len(ids)
            seen += len(ids)
        scheduler.step()

        if epoch in checkpoints:
            metrics = evaluate_raw(model, monitor)
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
        result["sparsity"] = model.sparsity_report()
        if epochs >= arch.PRUNE_END:
            for key, value in model.sparsity_report().items():
                if key != "overall" and abs(value - arch.FINAL_SPARSITY) > 0.002:
                    raise AssertionError(f"sparsity mismatch {key}={value}")
    return model, result


def aggregate(all_runs: List[Dict[str, object]], bank_names: Sequence[str]) -> Dict[str, object]:
    metrics = [
        "mae", "rmse", "median_sample_max_error", "p95_sample_max_error",
        "max_abs_error", "strict_0_1", "strict_0_5", "strict_1_0",
        "within_5_0", "relative_mae", "equation_residual_mean",
        "equation_residual_p95",
    ]
    out: Dict[str, object] = {}
    for label in ("PRUNED_RSNN", "ORDINARY_MLP"):
        out[label] = {}
        label_runs = [r for r in all_runs if r["label"] == label]
        for bank in bank_names:
            out[label][bank] = {}
            for metric in metrics:
                vals = [float(r["evaluations"][bank][metric]) for r in label_runs]
                out[label][bank][metric] = {
                    "mean": mean(vals),
                    "std": pstdev(vals) if len(vals) > 1 else 0.0,
                    "min": min(vals),
                    "max": max(vals),
                }
    wins: Dict[str, Dict[str, int]] = {}
    seeds = sorted({int(r["seed"]) for r in all_runs})
    for bank in bank_names:
        snn_wins = mlp_wins = ties = 0
        for seed in seeds:
            s = next(r for r in all_runs if r["label"] == "PRUNED_RSNN" and r["seed"] == seed)
            m = next(r for r in all_runs if r["label"] == "ORDINARY_MLP" and r["seed"] == seed)
            sm = s["evaluations"][bank]
            mm = m["evaluations"][bank]
            if sm["strict_0_1"] > mm["strict_0_1"] + 1e-12:
                snn_wins += 1
            elif mm["strict_0_1"] > sm["strict_0_1"] + 1e-12:
                mlp_wins += 1
            elif sm["mae"] < mm["mae"] - 1e-12:
                snn_wins += 1
            elif mm["mae"] < sm["mae"] - 1e-12:
                mlp_wins += 1
            else:
                ties += 1
        wins[bank] = {"PRUNED_RSNN": snn_wins, "ORDINARY_MLP": mlp_wins, "ties": ties}
    out["seed_wins_by_strict_0_1_then_mae"] = wins
    return out


def bank_manifest(bank: Bank) -> Dict[str, object]:
    c = bank.condition
    return {
        "size": len(bank.x),
        "attempts": bank.attempts,
        "target_abs_max": float(bank.y_raw.abs().max()),
        "target_abs_mean": float(bank.y_raw.abs().mean()),
        "condition_median": float(torch.quantile(c, 0.50)),
        "condition_p95": float(torch.quantile(c, 0.95)),
        "condition_max": float(c.max()),
        "sample_questions": bank.sample_questions,
    }


def preflight(dm_algebra) -> None:
    seen: set = set()
    train = build_bank(dm_algebra, name="PREFLIGHT_TRAIN", kind="train", count=48, seed=7001, global_seen=seen)
    test = build_bank(dm_algebra, name="PREFLIGHT_INTERP", kind="interpolate", count=32, seed=8001, global_seen=seen)
    stress = build_bank(dm_algebra, name="PREFLIGHT_STRESS12", kind="entropy12", count=32, seed=9001, global_seen=seen)
    torch.manual_seed(123)
    snn = arch.PrunedBrainSpikeNet()
    mlp = arch.OrdinaryMLP()
    if arch.count_parameters(snn) != 26880 or arch.count_parameters(mlp) != 26880:
        raise AssertionError("parameter budget mismatch")
    opt = torch.optim.AdamW(snn.parameters(), lr=arch.LR, weight_decay=arch.WEIGHT_DECAY)
    pred = snn(train.x[:16])
    loss = nn.SmoothL1Loss(beta=1.0)(pred, train.y_scaled[:16])
    loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in snn.parameters()):
        raise AssertionError("non-finite surrogate gradients")
    report = snn.prune_synapses(0.30, opt)
    snn.zero_pruned_gradients_()
    opt.step()
    snn.apply_masks_()
    if abs(report["overall"] - 0.30) > 0.001:
        raise AssertionError(report)
    _ = evaluate_raw(snn, test)
    _ = evaluate_raw(mlp, stress)
    print("DEEPMIND_PREFLIGHT_PASS", json.dumps({"sparsity": report, "train": bank_manifest(train), "test": bank_manifest(test), "stress": bank_manifest(stress)}, sort_keys=True), flush=True)


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
    parser.add_argument("--output-dir", default="deepmind-snn-stress-output")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    dm_algebra = prepare_deepmind()
    if args.preflight:
        preflight(dm_algebra)
        return

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) < 3:
        raise ValueError("harsh benchmark requires at least 3 independent model seeds")
    if args.epochs < arch.PRUNE_END:
        raise ValueError("epochs must reach the full 30% pruning schedule")

    seen: set = set()
    train = build_bank(dm_algebra, name="TRAIN_DEEPMIND", kind="train", count=args.train_size, seed=7301, global_seen=seen)
    interp = build_bank(dm_algebra, name="OFFICIAL_INTERPOLATE", kind="interpolate", count=args.eval_size, seed=8301, global_seen=seen)
    stress10 = build_bank(dm_algebra, name="STRESS_ENTROPY_10", kind="entropy10", count=args.eval_size, seed=9301, global_seen=seen)
    stress12 = build_bank(dm_algebra, name="STRESS_ENTROPY_12", kind="entropy12", count=args.eval_size, seed=10301, global_seen=seen)
    ill_pool = build_bank(dm_algebra, name="ILL_POOL_ENTROPY_12", kind="entropy12", count=args.ill_pool_size, seed=11301, global_seen=seen)
    ill_ids = torch.argsort(ill_pool.condition, descending=True)[: args.ill_size]
    ill = subset_bank(ill_pool, ill_ids, "ILL_CONDITIONED")
    banks = {
        interp.name: interp,
        stress10.name: stress10,
        stress12.name: stress12,
        ill.name: ill,
    }

    params_snn = arch.count_parameters(arch.PrunedBrainSpikeNet())
    params_mlp = arch.count_parameters(arch.OrdinaryMLP())
    if params_snn != params_mlp or params_snn != 26880:
        raise AssertionError(f"PARAMETER_MATCH_FAILED SNN={params_snn} MLP={params_mlp}")
    print(f"PARAMETER_MATCH SNN={params_snn} MLP={params_mlp}", flush=True)
    print("DATA_CONTRACT official google-deepmind/mathematics_dataset linear_2d only; project synthetic data=0", flush=True)
    print("STRESS_CONTRACT entropy10/entropy12 use official DeepMind generator implementation but are not named test_extra splits", flush=True)

    all_runs: List[Dict[str, object]] = []
    for seed in seeds:
        for label, cls in (("PRUNED_RSNN", arch.PrunedBrainSpikeNet), ("ORDINARY_MLP", arch.OrdinaryMLP)):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = cls()
            model, result = train_one(
                model, train, interp,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=seed,
                label=label,
            )
            result["evaluations"] = {name: evaluate_raw(model, bank) for name, bank in banks.items()}
            print(f"SEED_FINAL seed={seed} label={label} {json.dumps(result['evaluations'], sort_keys=True)}", flush=True)
            all_runs.append(result)
            del model

    bank_names = list(banks.keys())
    summary = aggregate(all_runs, bank_names)
    manifest = {
        "deepmind_repo": DM_REPO,
        "deepmind_commit": DM_COMMIT,
        "training_source": "official algebra.train linear_2d generator",
        "official_test_source": "official algebra.test linear_2d generator",
        "stress_note": "entropy10/12 are fixed-entropy stress banks from the official DeepMind generator implementation; mathematics_dataset exposes no named linear_2d test_extra split",
        "project_synthetic_examples": 0,
        "target_scale_used_only_for_optimization": TARGET_SCALE,
        "raw_metric_units": "original DeepMind solution units",
        "max_abs_target_filter": MAX_ABS_TARGET,
        "seeds": seeds,
        "epochs": args.epochs,
        "parameter_match": {"PRUNED_RSNN": params_snn, "ORDINARY_MLP": params_mlp},
        "banks": {
            train.name: bank_manifest(train),
            **{name: bank_manifest(bank) for name, bank in banks.items()},
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"manifest": manifest, "runs": all_runs, "aggregate": summary}, indent=2, sort_keys=True))
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    lines = [
        "DeepMind Pruned RSNN vs Ordinary MLP harsh benchmark",
        f"DeepMind commit={DM_COMMIT}",
        f"Seeds={seeds}",
        f"Train={len(train.x)} OfficialInterp={len(interp.x)} Stress10={len(stress10.x)} Stress12={len(stress12.x)} IllConditioned={len(ill.x)}",
        f"Params SNN={params_snn} MLP={params_mlp}",
        "Project synthetic examples=0",
    ]
    for bank in bank_names:
        lines.append(f"{bank} wins={summary['seed_wins_by_strict_0_1_then_mae'][bank]}")
        for label in ("PRUNED_RSNN", "ORDINARY_MLP"):
            a = summary[label][bank]
            lines.append(
                f"  {label}: strict0.1={a['strict_0_1']['mean']:.6f}±{a['strict_0_1']['std']:.6f} "
                f"strict1={a['strict_1_0']['mean']:.6f} mae={a['mae']['mean']:.6f} "
                f"p95={a['p95_sample_max_error']['mean']:.6f} residual={a['equation_residual_mean']['mean']:.6f}"
            )
    (out_dir / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    print("FINAL_AGGREGATE", json.dumps(summary, sort_keys=True), flush=True)
    print("DEEPMIND_HARSH_BENCHMARK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
