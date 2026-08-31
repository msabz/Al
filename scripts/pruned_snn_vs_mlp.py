#!/usr/bin/env python3
"""Controlled benchmark: pruned recurrent SNN vs parameter-matched ordinary MLP.

Task
----
Solve 2x2 linear systems from six coefficients:
    a*x + b*y = c
    d*x + e*y = f
Input:  [a,b,c,d,e,f] -> 6 values
Output: [x,y] -> 2 values

The SNN follows the requested architecture:
- hidden LIF layer: 160 neurons
- recurrent RSNN dynamics over T=25 steps
- beta=0.88, threshold=1.0
- hard binary spikes in forward pass
- fast-sigmoid surrogate derivative in backward pass, alpha=2.0
- subtractive soft reset
- output accumulated over time and averaged
- magnitude pruning from epoch 50 through 150 every 10 epochs
- final per-matrix sparsity 30%
- pruned gradients zeroed and pruned optimizer state frozen
- SmoothL1Loss(beta=1.0)
- AdamW(lr=0.003, weight_decay=1e-4)
- CosineAnnealingLR(T_max=300, eta_min=1e-5)

Baseline
--------
A dense ordinary ReLU MLP with the exact same dense parameter budget:
6 -> 160 -> 160 -> 2, no biases.
Both models therefore begin with exactly 26,880 trainable scalar weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


IN_DIM = 6
HIDDEN_DIM = 160
OUT_DIM = 2
TIME_STEPS = 25
DECAY = 0.88
THRESHOLD = 1.0
SURROGATE_ALPHA = 2.0
LR = 0.003
WEIGHT_DECAY = 1e-4
ETA_MIN = 1e-5
EPOCHS = 300
PRUNE_START = 50
PRUNE_END = 150
PRUNE_EVERY = 10
FINAL_SPARSITY = 0.30
STRICT_EPS = 0.10
DEVICE = torch.device("cpu")


class FastSigmoidSpike(torch.autograd.Function):
    """Hard threshold forward; fast-sigmoid surrogate derivative backward.

    Forward:  s = 1[x >= 0]
    Backward: ds/dx ~= 1 / (1 + alpha*|x|)^2
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = float(alpha)
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        surrogate = 1.0 / torch.square(1.0 + alpha * torch.abs(x))
        return grad_output * surrogate, None


def spike_fn(x: torch.Tensor) -> torch.Tensor:
    return FastSigmoidSpike.apply(x, SURROGATE_ALPHA)


class PrunedBrainSpikeNet(nn.Module):
    def __init__(
        self,
        in_dim: int = IN_DIM,
        hidden_dim: int = HIDDEN_DIM,
        out_dim: int = OUT_DIM,
        time_steps: int = TIME_STEPS,
        decay: float = DECAY,
        threshold: float = THRESHOLD,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.time_steps = time_steps
        self.decay = float(decay)
        self.threshold = float(threshold)

        self.W_in = nn.Parameter(torch.empty(hidden_dim, in_dim))
        self.W_rec = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.W_out = nn.Parameter(torch.empty(out_dim, hidden_dim))

        nn.init.normal_(self.W_in, mean=0.0, std=1.0 / math.sqrt(in_dim))
        nn.init.normal_(self.W_rec, mean=0.0, std=1.0 / math.sqrt(hidden_dim))
        nn.init.normal_(self.W_out, mean=0.0, std=1.0 / math.sqrt(hidden_dim))

        self.register_buffer("M_in", torch.ones_like(self.W_in))
        self.register_buffer("M_rec", torch.ones_like(self.W_rec))
        self.register_buffer("M_out", torch.ones_like(self.W_out))

    def masked_parameters(self):
        return (
            (self.W_in, self.M_in, "W_in"),
            (self.W_rec, self.M_rec, "W_rec"),
            (self.W_out, self.M_out, "W_out"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Masks are multiplied on every forward pass, so a pruned connection can
        # never contribute even if stale optimizer state exists.
        w_in = self.W_in * self.M_in
        w_rec = self.W_rec * self.M_rec
        w_out = self.W_out * self.M_out

        batch = x.shape[0]
        mem = x.new_zeros(batch, self.hidden_dim)
        spikes = x.new_zeros(batch, self.hidden_dim)
        out_acc = x.new_zeros(batch, self.out_dim)

        synaptic_input = F.linear(x, w_in)
        for _ in range(self.time_steps):
            recurrent_input = F.linear(spikes, w_rec)
            mem = self.decay * mem + synaptic_input + recurrent_input
            spikes = spike_fn(mem - self.threshold)
            mem = mem - spikes * self.threshold
            out_acc = out_acc + F.linear(spikes, w_out)

        return out_acc / float(self.time_steps)

    @torch.no_grad()
    def apply_masks_(self) -> None:
        for weight, mask, _ in self.masked_parameters():
            weight.mul_(mask)

    def zero_pruned_gradients_(self) -> None:
        for weight, mask, _ in self.masked_parameters():
            if weight.grad is not None:
                weight.grad.mul_(mask)

    @torch.no_grad()
    def _prune_one_to_target(
        self,
        weight: nn.Parameter,
        mask: torch.Tensor,
        target_sparsity: float,
        optimizer: torch.optim.Optimizer | None,
    ) -> None:
        target_sparsity = float(min(max(target_sparsity, 0.0), 1.0))
        total = mask.numel()
        target_pruned = int(round(total * target_sparsity))
        flat_mask = mask.view(-1)
        current_pruned = int((flat_mask == 0).sum().item())
        need = target_pruned - current_pruned
        if need <= 0:
            return

        active_idx = torch.nonzero(flat_mask > 0.5, as_tuple=False).flatten()
        active_abs = weight.detach().view(-1)[active_idx].abs()
        need = min(need, active_idx.numel())
        local = torch.topk(active_abs, k=need, largest=False, sorted=False).indices
        prune_idx = active_idx[local]
        flat_mask[prune_idx] = 0.0
        weight.mul_(mask)

        # Freeze Adam/AdamW momentum for newly pruned entries as well. Gradient
        # masking alone is insufficient because stale exp_avg can otherwise move
        # a parameter after pruning.
        if optimizer is not None:
            state = optimizer.state.get(weight, {})
            for value in state.values():
                if torch.is_tensor(value) and value.shape == weight.shape:
                    value.mul_(mask)

    @torch.no_grad()
    def prune_synapses(
        self,
        target_sparsity: float,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> Dict[str, float]:
        for weight, mask, _ in self.masked_parameters():
            self._prune_one_to_target(weight, mask, target_sparsity, optimizer)
        self.apply_masks_()
        return self.sparsity_report()

    @torch.no_grad()
    def sparsity_report(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        total_all = 0
        zeros_all = 0
        for _, mask, name in self.masked_parameters():
            total = mask.numel()
            zeros = int((mask == 0).sum().item())
            out[name] = zeros / float(total)
            total_all += total
            zeros_all += zeros
        out["overall"] = zeros_all / float(total_all)
        return out


class OrdinaryMLP(nn.Module):
    """Dense ordinary network with exactly the same 26,880 weights."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, HIDDEN_DIM, bias=False)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.fc3 = nn.Linear(HIDDEN_DIM, OUT_DIM, bias=False)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.normal_(self.fc3.weight, mean=0.0, std=1.0 / math.sqrt(HIDDEN_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


@dataclass
class Metrics:
    mae: float
    rmse: float
    strict_precision: float
    within_0_5: float
    within_1_0: float
    max_abs_error: float


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def generate_linear_systems(n: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate well-conditioned 2x2 systems with known solutions.

    x,y and coefficient magnitudes are bounded, then c/f are derived exactly.
    c and f are divided by 2 so every input channel is approximately in [-1,1].
    """
    rng = np.random.default_rng(seed)
    inputs = []
    targets = []
    while len(inputs) < n:
        xy = rng.uniform(-1.0, 1.0, size=2)
        a, b, d, e = rng.uniform(-1.0, 1.0, size=4)
        det = a * e - b * d
        if abs(det) < 0.25:
            continue
        c = a * xy[0] + b * xy[1]
        f = d * xy[0] + e * xy[1]
        inputs.append([a, b, c / 2.0, d, e, f / 2.0])
        targets.append(xy.tolist())
    return (
        torch.tensor(np.asarray(inputs), dtype=torch.float32),
        torch.tensor(np.asarray(targets), dtype=torch.float32),
    )


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 512) -> Metrics:
    model.eval()
    preds = []
    for start in range(0, len(x), batch_size):
        preds.append(model(x[start : start + batch_size].to(DEVICE)).cpu())
    pred = torch.cat(preds, dim=0)
    err = pred - y
    abs_err = err.abs()
    per_sample_max = abs_err.amax(dim=1)
    return Metrics(
        mae=float(abs_err.mean().item()),
        rmse=float(torch.sqrt(torch.mean(err * err)).item()),
        strict_precision=float((per_sample_max <= STRICT_EPS).float().mean().item()),
        within_0_5=float((per_sample_max <= 0.5).float().mean().item()),
        within_1_0=float((per_sample_max <= 1.0).float().mean().item()),
        max_abs_error=float(abs_err.max().item()),
    )


def pruning_target_for_epoch(epoch: int) -> float | None:
    if epoch < PRUNE_START or epoch > PRUNE_END or epoch % PRUNE_EVERY != 0:
        return None
    progress = (epoch - PRUNE_START) / float(PRUNE_END - PRUNE_START)
    return FINAL_SPARSITY * progress


def deterministic_batches(n: int, batch_size: int, seed: int, epoch: int) -> Iterable[torch.Tensor]:
    g = torch.Generator().manual_seed(seed * 100_003 + epoch)
    order = torch.randperm(n, generator=g)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def train_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    valid_x: torch.Tensor,
    valid_y: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    label: str,
) -> Dict[str, object]:
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=ETA_MIN)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    history = []
    start_time = time.perf_counter()

    checkpoints = {1, 10, 25, 50, 60, 80, 100, 120, 140, 150, 175, 200, 250, epochs}
    checkpoints = {e for e in checkpoints if 1 <= e <= epochs}

    for epoch in range(1, epochs + 1):
        if isinstance(model, PrunedBrainSpikeNet):
            target = pruning_target_for_epoch(epoch)
            if target is not None:
                report = model.prune_synapses(target, optimizer)
                print(f"PRUNE label={label} epoch={epoch} target={target:.4f} actual={report}", flush=True)

        model.train()
        running = 0.0
        seen = 0
        for ids in deterministic_batches(len(train_x), batch_size, seed, epoch):
            bx = train_x[ids].to(DEVICE)
            by = train_y[ids].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = loss_fn(pred, by)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss for {label} at epoch {epoch}")
            loss.backward()
            if isinstance(model, PrunedBrainSpikeNet):
                model.zero_pruned_gradients_()
            optimizer.step()
            if isinstance(model, PrunedBrainSpikeNet):
                model.apply_masks_()
            running += float(loss.detach().item()) * len(ids)
            seen += len(ids)

        scheduler.step()
        if epoch in checkpoints:
            metrics = evaluate(model, valid_x, valid_y)
            row = {
                "epoch": epoch,
                "train_loss": running / max(seen, 1),
                "lr": optimizer.param_groups[0]["lr"],
                "metrics": asdict(metrics),
            }
            if isinstance(model, PrunedBrainSpikeNet):
                row["sparsity"] = model.sparsity_report()
            history.append(row)
            print(f"CHECKPOINT label={label} {json.dumps(row, sort_keys=True)}", flush=True)

    seconds = time.perf_counter() - start_time
    final = evaluate(model, valid_x, valid_y)
    result: Dict[str, object] = {
        "label": label,
        "params": count_parameters(model),
        "train_seconds": seconds,
        "final": asdict(final),
        "history": history,
    }
    if isinstance(model, PrunedBrainSpikeNet):
        result["sparsity"] = model.sparsity_report()
        # The full 300-epoch run must finish at 30% sparsity in every matrix.
        if epochs >= PRUNE_END:
            report = model.sparsity_report()
            for name in ("W_in", "W_rec", "W_out"):
                max_rounding_error = 1.0 / dict(model.masked_parameters())[name].numel() if False else None
                if abs(report[name] - FINAL_SPARSITY) > 0.002:
                    raise AssertionError(f"final sparsity mismatch {name}: {report[name]}")
            # Strong freeze invariant: masked weights must be exactly zero.
            for weight, mask, name in model.masked_parameters():
                if torch.count_nonzero(weight.detach() * (1.0 - mask)).item() != 0:
                    raise AssertionError(f"pruned weights regrew in {name}")
    return result


def run_invariant_checks() -> None:
    torch.manual_seed(7)
    model = PrunedBrainSpikeNet(hidden_dim=32)
    x = torch.randn(8, IN_DIM, requires_grad=False)
    y = model(x)
    assert y.shape == (8, OUT_DIM)
    assert set(torch.unique(spike_fn(torch.tensor([-1.0, 0.0, 1.0]))).tolist()) <= {0.0, 1.0}

    z = torch.tensor([-0.5, 0.0, 0.5], requires_grad=True)
    spike_fn(z).sum().backward()
    assert torch.isfinite(z.grad).all() and float(z.grad[1]) > 0.0

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.prune_synapses(0.30, opt)
    report = model.sparsity_report()
    for name in ("W_in", "W_rec", "W_out"):
        assert abs(report[name] - 0.30) < 0.02, (name, report[name])

    pred = model(x)
    pred.square().mean().backward()
    model.zero_pruned_gradients_()
    for weight, mask, name in model.masked_parameters():
        if weight.grad is not None:
            assert torch.count_nonzero(weight.grad * (1.0 - mask)).item() == 0, name
    opt.step()
    model.apply_masks_()
    for weight, mask, name in model.masked_parameters():
        assert torch.count_nonzero(weight * (1.0 - mask)).item() == 0, name
    print("SNN_INVARIANT_CHECKS_PASS", report, flush=True)


def choose_winner(a: Dict[str, object], b: Dict[str, object]) -> str:
    # Primary metric is the requested strict precision. MAE is the tiebreaker.
    af = a["final"]
    bf = b["final"]
    if af["strict_precision"] > bf["strict_precision"] + 1e-12:
        return str(a["label"])
    if bf["strict_precision"] > af["strict_precision"] + 1e-12:
        return str(b["label"])
    if af["mae"] < bf["mae"]:
        return str(a["label"])
    if bf["mae"] < af["mae"]:
        return str(b["label"])
    return "TIE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--valid-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="snn-benchmark-output")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    run_invariant_checks()

    if args.smoke:
        args.epochs = 3
        args.train_size = 256
        args.valid_size = 128
        args.batch_size = 128

    train_x, train_y = generate_linear_systems(args.train_size, args.seed + 1)
    valid_x, valid_y = generate_linear_systems(args.valid_size, args.seed + 2)

    torch.manual_seed(args.seed + 10)
    snn = PrunedBrainSpikeNet()
    torch.manual_seed(args.seed + 20)
    mlp = OrdinaryMLP()

    snn_params = count_parameters(snn)
    mlp_params = count_parameters(mlp)
    if snn_params != 26_880 or mlp_params != 26_880 or snn_params != mlp_params:
        raise AssertionError(f"parameter budget mismatch: SNN={snn_params}, MLP={mlp_params}")
    print(f"PARAMETER_MATCH SNN={snn_params} MLP={mlp_params}", flush=True)
    print(
        "SNN_CONTRACT "
        f"hidden={HIDDEN_DIM} T={TIME_STEPS} decay={DECAY} threshold={THRESHOLD} "
        f"surrogate_alpha={SURROGATE_ALPHA} final_sparsity={FINAL_SPARSITY}",
        flush=True,
    )

    snn_result = train_model(
        snn, train_x, train_y, valid_x, valid_y,
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, label="PRUNED_RSNN",
    )
    mlp_result = train_model(
        mlp, train_x, train_y, valid_x, valid_y,
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, label="ORDINARY_MLP",
    )

    winner = choose_winner(snn_result, mlp_result)
    payload = {
        "task": "2x2 linear systems from six coefficients",
        "input_dim": IN_DIM,
        "output_dim": OUT_DIM,
        "train_size": args.train_size,
        "valid_size": args.valid_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "strict_epsilon": STRICT_EPS,
        "parameter_match": {"PRUNED_RSNN": snn_params, "ORDINARY_MLP": mlp_params},
        "PRUNED_RSNN": snn_result,
        "ORDINARY_MLP": mlp_result,
        "winner_by_strict_then_mae": winner,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    summary = (
        f"winner={winner}\n"
        f"PRUNED_RSNN={json.dumps(snn_result['final'], sort_keys=True)}\n"
        f"ORDINARY_MLP={json.dumps(mlp_result['final'], sort_keys=True)}\n"
        f"SNN_SPARSITY={json.dumps(snn_result.get('sparsity', {}), sort_keys=True)}\n"
        f"SNN_PARAMS={snn_params} MLP_PARAMS={mlp_params}\n"
    )
    (out_dir / "SUMMARY.txt").write_text(summary)
    print("FINAL_COMPARISON", json.dumps(payload, sort_keys=True), flush=True)
    print("BENCHMARK_COMPLETE", winner, flush=True)


if __name__ == "__main__":
    main()
