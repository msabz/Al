#!/usr/bin/env python3
"""Controlled INT8 ablation for the adopted P30 RSNN.

Goal
----
Compare the adopted supervised 30%-pruned RSNN before and after INT8 weight
quantization while keeping the DeepMind benchmark contract unchanged.

Variants
--------
1) P30_FP32_BASELINE
   Existing supervised P30 recipe, FP32 weights.
2) P30_INT8_PTQ
   Exact trained baseline quantized post-training to symmetric per-output-channel
   INT8 weights. This isolates pure quantization damage/benefit with no retraining.
3) P30_INT8_QAT
   Same initialization, data, batches, optimizer and pruning schedule as baseline.
   Fake-quantized INT8 weights are enabled only after the 30% pruning schedule is
   complete (epoch 151 onward), then the final model is exported to real INT8
   tensors plus FP32 per-channel scales for evaluation.

Important scope
---------------
- Weight-only INT8. Activations, membrane state and accumulators remain FP32.
- The exported INT8 model stores actual torch.int8 weight tensors.
- Evaluation dequantizes those stored INT8 weights to reproduce the quantized
  numerical model exactly. Therefore this experiment measures accuracy and
  storage effects, not an INT8-kernel speedup claim.
- Same pinned Google DeepMind mathematics_dataset linear_2d banks as the prior
  harsh benchmark; no project synthetic examples.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pruned_snn_vs_mlp as arch
import deepmind_pruned_snn_stress as dm
import deepmind_rsnn_pruning_sweep as sweep

DEVICE = torch.device("cpu")
FINAL_SPARSITY = 0.30
QAT_START_EPOCH = arch.PRUNE_END + 1
LABEL_FP32 = "P30_FP32_BASELINE"
LABEL_PTQ = "P30_INT8_PTQ"
LABEL_QAT = "P30_INT8_QAT"
LABELS = (LABEL_FP32, LABEL_PTQ, LABEL_QAT)
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


def clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def assert_same_initialization(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], context: str) -> None:
    for name in ("W_in", "W_rec", "W_out", "M_in", "M_rec", "M_out"):
        if not torch.equal(a[name], b[name]):
            raise AssertionError(f"initialization mismatch {context} param={name}")


def symmetric_per_channel_quantize(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize rows (output channels) to signed INT8 [-127,127]."""
    w = weight.detach().to(torch.float32)
    max_abs = w.abs().amax(dim=1, keepdim=True)
    scale = torch.where(max_abs > 0.0, max_abs / 127.0, torch.ones_like(max_abs))
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q.contiguous(), scale.squeeze(1).to(torch.float32).contiguous()


def fake_quant_per_channel_ste(weight: torch.Tensor) -> torch.Tensor:
    """Per-output-channel symmetric INT8 fake quantization with STE."""
    max_abs = weight.detach().abs().amax(dim=1, keepdim=True)
    scale = torch.where(max_abs > 0.0, max_abs / 127.0, torch.ones_like(max_abs))
    qdq = torch.clamp(torch.round(weight / scale), -127, 127) * scale
    return weight + (qdq - weight).detach()


class QATP30RSNN(arch.PrunedBrainSpikeNet):
    """Same RSNN, with switchable weight fake-quantization for QAT."""

    def __init__(self) -> None:
        super().__init__()
        self.quant_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_in = self.W_in * self.M_in
        w_rec = self.W_rec * self.M_rec
        w_out = self.W_out * self.M_out
        if self.quant_enabled:
            w_in = fake_quant_per_channel_ste(w_in)
            w_rec = fake_quant_per_channel_ste(w_rec)
            w_out = fake_quant_per_channel_ste(w_out)

        batch = x.shape[0]
        mem = x.new_zeros(batch, self.hidden_dim)
        spikes = x.new_zeros(batch, self.hidden_dim)
        out_acc = x.new_zeros(batch, self.out_dim)
        synaptic_input = F.linear(x, w_in)
        for _ in range(self.time_steps):
            recurrent_input = F.linear(spikes, w_rec)
            mem = self.decay * mem + synaptic_input + recurrent_input
            spikes = arch.spike_fn(mem - self.threshold)
            mem = mem - spikes * self.threshold
            out_acc = out_acc + F.linear(spikes, w_out)
        return out_acc / float(self.time_steps)


class Int8WeightRSNN(nn.Module):
    """Inference model backed by real INT8 weight tensors and FP32 scales."""

    def __init__(self, source: arch.PrunedBrainSpikeNet) -> None:
        super().__init__()
        self.in_dim = source.in_dim
        self.hidden_dim = source.hidden_dim
        self.out_dim = source.out_dim
        self.time_steps = source.time_steps
        self.decay = float(source.decay)
        self.threshold = float(source.threshold)

        stats: Dict[str, Dict[str, float]] = {}
        for name, weight, mask in (
            ("W_in", source.W_in, source.M_in),
            ("W_rec", source.W_rec, source.M_rec),
            ("W_out", source.W_out, source.M_out),
        ):
            masked = (weight.detach() * mask.detach()).to(torch.float32)
            q, scale = symmetric_per_channel_quantize(masked)
            self.register_buffer(f"q_{name}", q)
            self.register_buffer(f"scale_{name}", scale)
            self.register_buffer(f"mask_{name}", mask.detach().to(torch.bool).clone())

            recon = q.to(torch.float32) * scale[:, None]
            active = mask.detach() > 0.5
            if bool(active.any()):
                err = (recon[active] - masked[active]).abs()
                active_mae = float(err.mean().item())
                active_max = float(err.max().item())
                active_count = int(active.sum().item())
                extra_zero = int(((q == 0) & active).sum().item())
            else:
                active_mae = active_max = 0.0
                active_count = extra_zero = 0
            stats[name] = {
                "active_count": active_count,
                "active_quant_mae": active_mae,
                "active_quant_max_abs": active_max,
                "extra_zeroed_active": extra_zero,
                "effective_zero_fraction": float((q == 0).float().mean().item()),
            }
        self.quant_stats = stats

    def _dequant(self, name: str) -> torch.Tensor:
        q = getattr(self, f"q_{name}")
        scale = getattr(self, f"scale_{name}")
        return q.to(torch.float32) * scale[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_in = self._dequant("W_in")
        w_rec = self._dequant("W_rec")
        w_out = self._dequant("W_out")
        batch = x.shape[0]
        mem = x.new_zeros(batch, self.hidden_dim)
        spikes = x.new_zeros(batch, self.hidden_dim)
        out_acc = x.new_zeros(batch, self.out_dim)
        synaptic_input = F.linear(x, w_in)
        for _ in range(self.time_steps):
            recurrent_input = F.linear(spikes, w_rec)
            mem = self.decay * mem + synaptic_input + recurrent_input
            spikes = arch.spike_fn(mem - self.threshold)
            mem = mem - spikes * self.threshold
            out_acc = out_acc + F.linear(spikes, w_out)
        return out_acc / float(self.time_steps)

    def storage_report(self) -> Dict[str, object]:
        q_bytes = sum(getattr(self, f"q_{n}").numel() for n in ("W_in", "W_rec", "W_out"))
        scale_bytes = sum(getattr(self, f"scale_{n}").numel() * 4 for n in ("W_in", "W_rec", "W_out"))
        mask_bits = sum(getattr(self, f"mask_{n}").numel() for n in ("W_in", "W_rec", "W_out"))
        mask_bitmap_bytes = int(math.ceil(mask_bits / 8.0))
        active = sum(int(v["active_count"]) for v in self.quant_stats.values())
        fp32_dense_bytes = q_bytes * 4
        fp32_p30_packed_bytes = active * 4 + mask_bitmap_bytes
        int8_dense_plus_scales = q_bytes + scale_bytes
        int8_p30_packed_bytes = active + mask_bitmap_bytes + scale_bytes
        return {
            "dense_fp32_weight_bytes": fp32_dense_bytes,
            "dense_int8_weight_bytes": q_bytes,
            "per_channel_scale_bytes": scale_bytes,
            "dense_int8_plus_scales_bytes": int8_dense_plus_scales,
            "dense_int8_plus_scales_reduction_pct": 100.0 * (1.0 - int8_dense_plus_scales / fp32_dense_bytes),
            "logical_active_weights": active,
            "bitmap_bytes_if_sparse_packed": mask_bitmap_bytes,
            "p30_fp32_active_plus_bitmap_bytes": fp32_p30_packed_bytes,
            "p30_int8_active_plus_bitmap_plus_scales_bytes": int8_p30_packed_bytes,
            "p30_packed_reduction_vs_p30_fp32_pct": 100.0 * (1.0 - int8_p30_packed_bytes / fp32_p30_packed_bytes),
            "p30_packed_reduction_vs_original_dense_fp32_pct": 100.0 * (1.0 - int8_p30_packed_bytes / fp32_dense_bytes),
        }


def train_qat_variant(
    model: QATP30RSNN,
    train: dm.Bank,
    monitor: dm.Bank,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Tuple[QATP30RSNN, Dict[str, object]]:
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arch.LR, weight_decay=arch.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=arch.ETA_MIN)
    criterion = nn.SmoothL1Loss(beta=1.0)
    checkpoints = {1, 50, 100, 150, QAT_START_EPOCH, 200, 250, epochs}
    checkpoints = {e for e in checkpoints if 1 <= e <= epochs}
    history: List[Dict[str, object]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        target = sweep.pruning_target(epoch, FINAL_SPARSITY)
        if target is not None:
            report = model.prune_synapses(target, optimizer)
            print(
                f"PRUNE seed={seed} label={LABEL_QAT} epoch={epoch} target={target:.4f} "
                f"actual={json.dumps(report, sort_keys=True)}",
                flush=True,
            )

        model.quant_enabled = epoch >= QAT_START_EPOCH
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
                raise RuntimeError(f"non-finite QAT loss seed={seed} epoch={epoch}")
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
                "quant_enabled": model.quant_enabled,
                "official_interpolate": dm.evaluate_raw(model, monitor),
                "sparsity": model.sparsity_report(),
            }
            history.append(row)
            print(f"CHECKPOINT seed={seed} label={LABEL_QAT} {json.dumps(row, sort_keys=True)}", flush=True)

    elapsed = time.perf_counter() - started
    model.quant_enabled = True
    report = model.sparsity_report()
    for key in ("W_in", "W_rec", "W_out", "overall"):
        if abs(float(report[key]) - FINAL_SPARSITY) > 0.002:
            raise AssertionError(f"QAT final sparsity mismatch seed={seed} {key}={report[key]}")
    return model, {
        "seed": seed,
        "label": LABEL_QAT,
        "training": "supervised_qat",
        "qat_start_epoch": QAT_START_EPOCH,
        "params": arch.count_parameters(model),
        "active_weights": int(round(arch.count_parameters(model) * (1.0 - FINAL_SPARSITY))),
        "train_seconds": elapsed,
        "sparsity": report,
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
        "quantization_scope": "weight-only symmetric per-output-channel INT8; activations/membrane/accumulators FP32",
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


def aggregate_by_label(runs: List[Dict[str, object]], bank_names: Sequence[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    seeds = sorted({int(r["seed"]) for r in runs})
    for label in LABELS:
        rows = [r for r in runs if r["label"] == label]
        if len(rows) != len(seeds):
            raise AssertionError(f"missing rows label={label}: {len(rows)} != {len(seeds)}")
        entry: Dict[str, object] = {
            "params": int(rows[0].get("params", 0)),
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
        raise AssertionError(f"seed mismatch {reference} vs {candidate}")
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


def preflight() -> None:
    set_seed(1234)
    source = arch.PrunedBrainSpikeNet()
    opt = torch.optim.AdamW(source.parameters(), lr=arch.LR, weight_decay=arch.WEIGHT_DECAY)
    report = source.prune_synapses(FINAL_SPARSITY, opt)
    if abs(float(report["overall"]) - FINAL_SPARSITY) > 0.002:
        raise AssertionError(report)
    exported = Int8WeightRSNN(source)
    for n in ("W_in", "W_rec", "W_out"):
        q = getattr(exported, f"q_{n}")
        if q.dtype != torch.int8:
            raise AssertionError(f"{n} not int8")
        mask = getattr(exported, f"mask_{n}")
        if bool((q[~mask] != 0).any()):
            raise AssertionError(f"pruned zeros not preserved in {n}")

    qat = QATP30RSNN()
    qat.load_state_dict(source.state_dict(), strict=True)
    qat.quant_enabled = True
    exported_qat = Int8WeightRSNN(qat)
    x = torch.randn(17, arch.IN_DIM)
    qat.eval(); exported_qat.eval()
    with torch.no_grad():
        diff = float((qat(x) - exported_qat(x)).abs().max().item())
    if diff > 2e-6:
        raise AssertionError(f"fake-quant/export mismatch max_abs={diff}")
    storage = exported.storage_report()
    if float(storage["dense_int8_plus_scales_reduction_pct"]) < 70.0:
        raise AssertionError(storage)
    print(f"P30_INT8_PREFLIGHT_PASS max_export_diff={diff:.3e} storage={json.dumps(storage, sort_keys=True)}", flush=True)


def make_summary(aggregate: Dict[str, object], paired: Dict[str, object], storage: Dict[str, object]) -> str:
    lines = [
        "P30 RSNN INT8 ABLATION — DEEPMIND linear_2d",
        "",
        "Contract: same P30 supervised RSNN recipe; 5 paired seeds; 300 epochs; same DeepMind banks.",
        "INT8 scope: weight-only symmetric per-output-channel. Activations/membrane/accumulators remain FP32.",
        f"QAT starts at epoch {QAT_START_EPOCH}, after 30% pruning is complete.",
        "",
        "Storage payload:",
        json.dumps(storage, sort_keys=True),
        "",
    ]
    official = "OFFICIAL_INTERPOLATE"
    for label in LABELS:
        row = aggregate[label]
        b = row["banks"][official]
        lines.append(
            f"{label}: MAE={b['mae']['mean']:.8f} RMSE={b['rmse']['mean']:.8f} "
            f"strict1={100.0*b['strict_1_0']['mean']:.4f}% within5={100.0*b['within_5_0']['mean']:.4f}% "
            f"train_s={row['train_seconds']['mean']:.3f}"
        )
    lines += ["", "Paired comparisons vs FP32 baseline:"]
    for label in (LABEL_PTQ, LABEL_QAT):
        for bank, row in paired[f"{label}_vs_{LABEL_FP32}"].items():
            lines.append(
                f"{label} {bank}: MAE_delta={row['mae_change_pct_mean']:+.4f}% "
                f"strict1_delta={row['strict1_change_pp_mean']:+.4f}pp "
                f"within5_delta={row['within5_change_pp_mean']:+.4f}pp "
                f"wins={row['mae_wins']}/{row['mae_wins']+row['mae_losses']}"
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
    parser.add_argument("--output-dir", default="deepmind-p30-int8-ablation-output")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    if args.preflight:
        preflight()
        return
    if args.epochs < QAT_START_EPOCH:
        raise ValueError(f"epochs must be >= {QAT_START_EPOCH} to cover pruning then QAT")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) < 3:
        raise ValueError("at least 3 seeds required")

    train, banks, manifest = build_banks(args)
    params = arch.count_parameters(arch.PrunedBrainSpikeNet())
    active = int(round(params * (1.0 - FINAL_SPARSITY)))
    if params != 26880 or active != 18816:
        raise AssertionError(f"parameter contract changed params={params} active={active}")
    print(f"INT8_PARAMETER_CONTRACT params={params} P30_active={active}", flush=True)
    print("INT8_FAIRNESS_CONTRACT baseline and QAT share initialization/data/batches/optimizer/pruning; only INT8 fake quantization after epoch 150 differs", flush=True)
    print("INT8_PTQ_CONTROL exact FP32 trained baseline quantized with no retraining", flush=True)
    print("DATA_CONTRACT pinned google-deepmind/mathematics_dataset linear_2d; project synthetic data=0", flush=True)

    all_runs: List[Dict[str, object]] = []
    first_storage: Dict[str, object] | None = None
    for seed in seeds:
        set_seed(seed)
        base_model = arch.PrunedBrainSpikeNet()
        base_initial = clone_state(base_model)

        set_seed(seed)
        qat_model = QATP30RSNN()
        qat_initial = clone_state(qat_model)
        assert_same_initialization(base_initial, qat_initial, f"seed={seed}")

        base_model, base_result = sweep.train_variant(
            base_model,
            train,
            banks["OFFICIAL_INTERPOLATE"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=seed,
            final_sparsity=FINAL_SPARSITY,
        )
        base_result["label"] = LABEL_FP32
        base_result["training"] = "supervised_fp32"
        attach_evaluations(base_result, base_model, banks)
        all_runs.append(base_result)

        ptq_started = time.perf_counter()
        ptq_model = Int8WeightRSNN(base_model).to(DEVICE)
        ptq_seconds = time.perf_counter() - ptq_started
        storage = ptq_model.storage_report()
        if first_storage is None:
            first_storage = storage
        ptq_result: Dict[str, object] = {
            "seed": seed,
            "label": LABEL_PTQ,
            "training": "post_training_quantization",
            "params": params,
            "active_weights": active,
            "train_seconds": ptq_seconds,
            "storage": storage,
            "quant_stats": ptq_model.quant_stats,
        }
        attach_evaluations(ptq_result, ptq_model, banks)
        all_runs.append(ptq_result)

        qat_model, qat_result = train_qat_variant(
            qat_model,
            train,
            banks["OFFICIAL_INTERPOLATE"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=seed,
        )
        qat_export = Int8WeightRSNN(qat_model).to(DEVICE)
        qat_result["storage"] = qat_export.storage_report()
        qat_result["quant_stats"] = qat_export.quant_stats
        attach_evaluations(qat_result, qat_export, banks)
        all_runs.append(qat_result)

    bank_names = list(banks.keys())
    agg = aggregate_by_label(all_runs, bank_names)
    paired = {
        f"{LABEL_PTQ}_vs_{LABEL_FP32}": paired_compare(all_runs, LABEL_FP32, LABEL_PTQ, bank_names),
        f"{LABEL_QAT}_vs_{LABEL_FP32}": paired_compare(all_runs, LABEL_FP32, LABEL_QAT, bank_names),
        f"{LABEL_QAT}_vs_{LABEL_PTQ}": paired_compare(all_runs, LABEL_PTQ, LABEL_QAT, bank_names),
    }
    if first_storage is None:
        raise AssertionError("missing storage report")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "P30_RSNN_INT8_ABLATION",
        "device": str(DEVICE),
        "seeds": seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "final_sparsity": FINAL_SPARSITY,
        "qat_start_epoch": QAT_START_EPOCH,
        "quantization": {
            "weight_dtype": "int8",
            "scheme": "symmetric per-output-channel",
            "range": [-127, 127],
            "activations": "fp32",
            "membrane": "fp32",
            "accumulators": "fp32",
        },
        "storage": first_storage,
        "runs": all_runs,
        "aggregate": agg,
        "paired": paired,
    }
    (output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    summary = make_summary(agg, paired, first_storage)
    (output / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary, flush=True)


if __name__ == "__main__":
    main()
