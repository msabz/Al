#!/usr/bin/env python3
"""CPU-only controlled test of the six-branch 'onion' architecture.

Data contract: ONLY official google-deepmind/mathematics_dataset polynomial_roots
examples through the already-audited v6 bank builder. No project synthetic data.

Models:
  A) parameter-matched ordinary MLP.
  B) parameter-matched stabilized MLP using the same invertible I/O compressor.
  C) OnionManifoldNet:
       - input compressor: c(x)=asinh(x)/4
       - six symmetric branches
       - three manifold-weight layers per branch
       - initially dormant per-channel inter-layer links
       - correctness-driven reward/punishment gate regularizer
       - backward-only aggregation using the same compressor derivative
       - output inverse: c^-1(y)=sinh(4y)

Middle-layer effective scalar weights are projections of trainable points that
remain on one of four infinite-solution manifolds.
"""
import copy
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpu_v6_manifold_lattice_deepmind as base

OFFICIAL_DATA_CONTRACT = "google-deepmind/mathematics_dataset polynomial_roots only"
SEEDS = (11, 22, 33)
STEPS = 5000
BATCH = 192
LR = 1.2e-3
WEIGHT_DECAY = 1e-5
CHECKPOINTS = {1, 100, 250, 500, 1000, 2000, 3000, 4000, 5000}
BRANCHES = 6
DEPTH = 3
WIDTH = 7
COMPRESS_DIV = 4.0
ROOT_CODE_BOUND = 0.50
GATE_INIT_LOGIT = -4.0
GATE_REWARD_WEIGHT = 0.03
DEVICE = torch.device("cpu")


def quiet(x):
    """Invertible number compressor used at input and by backward-only aggregation."""
    return torch.asinh(x) / COMPRESS_DIV


def unquiet(y):
    """Exact inverse of quiet()."""
    return torch.sinh(COMPRESS_DIV * y)


class ParamMatchedMLP(nn.Module):
    """Ordinary scalar-weight baseline near the onion parameter budget."""
    def __init__(self, hidden=114):
        super().__init__()
        self.fc1 = nn.Linear(base.FEATURES, hidden)
        self.fc2 = nn.Linear(hidden, base.ROOT_SLOTS * 2)

    def forward(self, x, degree):
        del degree
        y = self.fc2(torch.tanh(self.fc1(x)))
        return y[:, :base.ROOT_SLOTS], y[:, base.ROOT_SLOTS:]


class StabilizedMLP(nn.Module):
    """Control for the input quieting/output inverse without onion topology."""
    def __init__(self, hidden=114):
        super().__init__()
        self.fc1 = nn.Linear(base.FEATURES, hidden)
        self.fc2 = nn.Linear(hidden, base.ROOT_SLOTS * 2)

    def forward(self, x, degree):
        del degree
        q = quiet(x)
        y = self.fc2(torch.tanh(self.fc1(q)))
        root_code = ROOT_CODE_BOUND * torch.tanh(y[:, :base.ROOT_SLOTS])
        roots = unquiet(root_code)
        return roots, y[:, base.ROOT_SLOTS:]


class ManifoldLinear(nn.Module):
    """No free scalar matrix: every effective weight comes from a manifold point."""
    def __init__(self, dim, family_offset):
        super().__init__()
        self.dim = dim
        self.theta = nn.Parameter(torch.randn(dim, dim) * 0.35)
        self.phi = nn.Parameter(torch.randn(dim, dim) * 0.35)
        self.bias = nn.Parameter(torch.zeros(dim))
        oi = torch.arange(dim)[:, None]
        ii = torch.arange(dim)[None, :]
        self.register_buffer("family", (oi + 2 * ii + family_offset) % 4, persistent=False)

    def manifold_point(self):
        theta = self.theta
        phi = self.phi
        cp = torch.cos(phi)
        sphere = torch.stack((
            torch.cos(theta) * cp,
            torch.sin(theta) * cp,
            torch.sin(phi),
        ), dim=-1)
        cylinder = torch.stack((
            torch.cos(theta),
            torch.sin(theta),
            torch.tanh(phi),
        ), dim=-1)
        u = torch.tanh(theta)
        v = torch.tanh(phi)
        saddle = torch.stack((u, v, u * v), dim=-1)
        wave = torch.stack((
            u,
            v,
            torch.tanh(torch.sin(math.pi * u) + torch.cos(math.pi * v)),
        ), dim=-1)
        f = self.family
        p = torch.where((f == 0)[..., None], sphere, cylinder)
        p = torch.where((f == 2)[..., None], saddle, p)
        p = torch.where((f == 3)[..., None], wave, p)
        return p

    def effective_weight(self):
        p = self.manifold_point()
        return (0.55 * p[..., 0] + 0.30 * p[..., 1] + 0.15 * p[..., 2]) / math.sqrt(self.dim)

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)


class OnionManifoldNet(nn.Module):
    """Six symmetric 3-layer chains with dormant correctness-controlled links."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            ManifoldLinear(WIDTH, family_offset=(chain * DEPTH + depth) % 4)
            for chain in range(BRANCHES)
            for depth in range(DEPTH)
        ])
        self.gate_logits = nn.Parameter(
            torch.full((BRANCHES, DEPTH - 1, WIDTH), GATE_INIT_LOGIT)
        )
        self.readout = nn.Linear(WIDTH, base.ROOT_SLOTS * 2)

    def _layer(self, chain, depth):
        return self.layers[chain * DEPTH + depth]

    def gate_values(self):
        return torch.sigmoid(self.gate_logits)

    def forward(self, x, degree):
        del degree
        q = quiet(x)
        branch_outputs = []
        gates = self.gate_values()
        for chain in range(BRANCHES):
            h0 = torch.tanh(self._layer(chain, 0)(q))
            proposal1 = torch.tanh(self._layer(chain, 1)(h0))
            h1 = h0 + gates[chain, 0] * proposal1
            proposal2 = torch.tanh(self._layer(chain, 2)(h1))
            h2 = h1 + gates[chain, 1] * proposal2
            branch_outputs.append(h2)

        branches = torch.stack(branch_outputs, dim=1)
        plain_mean = branches.mean(dim=1)

        # Backward-only aggregation: forward value equals plain_mean exactly,
        # while backward follows quiet'(plain_mean), the same equation as input.
        compressed = quiet(plain_mean)
        aggregate = plain_mean.detach() + compressed - compressed.detach()

        y = self.readout(aggregate)
        root_code = ROOT_CODE_BOUND * torch.tanh(y[:, :base.ROOT_SLOTS])
        roots = unquiet(root_code)
        return roots, y[:, base.ROOT_SLOTS:]

    @torch.no_grad()
    def gate_stats(self):
        g = self.gate_values().flatten()
        return {
            "mean_gate": float(g.mean()),
            "median_gate": float(g.median()),
            "links_gt_0.10": int((g > 0.10).sum()),
            "links_gt_0.25": int((g > 0.25).sum()),
            "links_gt_0.50": int((g > 0.50).sum()),
            "max_gate": float(g.max()),
        }

    @torch.no_grad()
    def manifold_stats(self, initial_points):
        points = torch.stack([layer.manifold_point() for layer in self.layers], dim=0)
        d = torch.linalg.vector_norm(points - initial_points, dim=-1)
        return {
            "mean_point_displacement": float(d.mean()),
            "max_point_displacement": float(d.max()),
        }


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def correctness_reward(primary_loss):
    return torch.sigmoid((0.30 - primary_loss.detach()) * 12.0)


def onion_loss(model, x, roots, degree):
    primary = base.loss_fn(model, x, roots, degree)
    if not isinstance(model, OnionManifoldNet):
        return primary, float("nan")
    reward = correctness_reward(primary)
    gates = model.gate_values()
    target = torch.ones_like(gates) * reward
    reward_penalty = F.binary_cross_entropy(gates, target)
    total = primary + GATE_REWARD_WEIGHT * reward_penalty
    return total, float(reward)


def make_fresh(kind):
    if kind == "MLP":
        return ParamMatchedMLP()
    if kind == "STABILIZED":
        return StabilizedMLP()
    if kind == "ONION":
        return OnionManifoldNet()
    raise ValueError(kind)


def train_one(kind, train, valid, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = make_fresh(kind).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    batch_rng = torch.Generator().manual_seed(100000 + seed)
    initial_gates = model.gate_stats() if kind == "ONION" else None
    initial_points = None
    if kind == "ONION":
        initial_points = torch.stack(
            [layer.manifold_point().detach().clone() for layer in model.layers], dim=0
        )

    best_score = float("inf")
    best_state = None
    curve = {}
    for step in range(1, STEPS + 1):
        ids = torch.randint(0, len(train.features), (BATCH,), generator=batch_rng)
        opt.zero_grad(set_to_none=True)
        loss, reward = onion_loss(
            model,
            train.features[ids],
            train.roots[ids],
            train.degree[ids],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step in CHECKPOINTS:
            metrics = base.evaluate(model, valid)
            gates = model.gate_stats() if kind == "ONION" else None
            curve[step] = metrics
            score = metrics["mae"] + 50.0 * (1.0 - metrics["count"]) + 20.0 * (1.0 - metrics["within1"])
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"{kind} seed={seed} step={step:4d} loss={float(loss.detach()):.6f} "
                f"reward={reward:.4f} VALID={metrics} GATES={gates}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    final = base.evaluate(model, valid)
    final_gates = model.gate_stats() if kind == "ONION" else None
    geometry = model.manifold_stats(initial_points) if kind == "ONION" else None
    return final, curve, initial_gates, final_gates, geometry


def main():
    print("DEVICE=cpu")
    print("DATA_CONTRACT", OFFICIAL_DATA_CONTRACT)
    print("ONION_CONTRACT branches=6 depth=3 dormant_links=reward_punish backward_only_aggregation")
    dm = base.prepare_official_deepmind()
    print("Building deterministic official DeepMind banks...", flush=True)
    train = base.build_bank(dm, base.TRAIN_COUNT, 7301)
    valid = base.build_bank(dm, base.VALID_COUNT, 8301)
    print("train degree distribution", {d: int((train.degree == d).sum()) for d in range(2, 6)})
    print("valid degree distribution", {d: int((valid.degree == d).sum()) for d in range(2, 6)})

    params = {k: parameter_count(make_fresh(k)) for k in ("MLP", "STABILIZED", "ONION")}
    print("PARAMS", params)
    ref = params["MLP"]
    for k, n in params.items():
        if abs(n - ref) / ref > 0.05:
            raise SystemExit(f"parameter budget mismatch >5% for {k}: {n} vs {ref}")

    rows = []
    for seed in SEEDS:
        results = {}
        curves = {}
        diagnostics = {}
        for kind in ("MLP", "STABILIZED", "ONION"):
            result, curve, gi, gf, geo = train_one(kind, train, valid, seed)
            results[kind] = result
            curves[kind] = curve
            diagnostics[kind] = {"gate_initial": gi, "gate_final": gf, "geometry": geo}
        rows.append((seed, results, diagnostics))
        print(f"SEED_SUMMARY seed={seed} RESULTS={results} DIAGNOSTICS={diagnostics}", flush=True)
        print("LATE_CURVE", seed, {
            s: {
                "mlp_mae": curves["MLP"][s]["mae"],
                "stabilized_mae": curves["STABILIZED"][s]["mae"],
                "onion_mae": curves["ONION"][s]["mae"],
                "mlp_count": curves["MLP"][s]["count"],
                "stabilized_count": curves["STABILIZED"][s]["count"],
                "onion_count": curves["ONION"][s]["count"],
                "mlp_within1": curves["MLP"][s]["within1"],
                "stabilized_within1": curves["STABILIZED"][s]["within1"],
                "onion_within1": curves["ONION"][s]["within1"],
            }
            for s in sorted(CHECKPOINTS)
        }, flush=True)

    def avg(kind, key):
        return float(np.mean([results[kind][key] for _, results, _ in rows]))

    summary = {
        kind: {key: avg(kind, key) for key in ("count", "mae", "within1", "merge")}
        for kind in ("MLP", "STABILIZED", "ONION")
    }
    summary["delta_onion_vs_mlp"] = {
        "count": summary["ONION"]["count"] - summary["MLP"]["count"],
        "mae": summary["ONION"]["mae"] - summary["MLP"]["mae"],
        "within1": summary["ONION"]["within1"] - summary["MLP"]["within1"],
        "merge": summary["ONION"]["merge"] - summary["MLP"]["merge"],
    }
    summary["delta_onion_vs_stabilized"] = {
        "count": summary["ONION"]["count"] - summary["STABILIZED"]["count"],
        "mae": summary["ONION"]["mae"] - summary["STABILIZED"]["mae"],
        "within1": summary["ONION"]["within1"] - summary["STABILIZED"]["within1"],
        "merge": summary["ONION"]["merge"] - summary["STABILIZED"]["merge"],
    }

    wins_mlp = sum(
        results["ONION"]["mae"] < results["MLP"]["mae"]
        and results["ONION"]["count"] >= results["MLP"]["count"]
        for _, results, _ in rows
    )
    wins_stable = sum(
        results["ONION"]["mae"] < results["STABILIZED"]["mae"]
        and results["ONION"]["count"] >= results["STABILIZED"]["count"]
        for _, results, _ in rows
    )

    print("FINAL_SUMMARY", summary, flush=True)
    print(f"ONION_WINS_VS_MLP={wins_mlp}/{len(SEEDS)}")
    print(f"ONION_WINS_VS_STABILIZED={wins_stable}/{len(SEEDS)}")

    clear = (
        wins_mlp >= 2
        and summary["delta_onion_vs_mlp"]["mae"] <= -2.0
        and summary["delta_onion_vs_mlp"]["within1"] >= 0.01
    )
    if clear:
        print("V8_ONION_MANIFOLD_DEEPMIND_CLEAR_SIGNAL")
    else:
        print("V8_ONION_MANIFOLD_DEEPMIND_NO_CLEAR_SIGNAL")


if __name__ == "__main__":
    main()
