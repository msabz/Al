#!/usr/bin/env python3
"""CPU-only controlled test of geometry-driven intersection routing.

Data contract: ONLY official google-deepmind/mathematics_dataset polynomial_roots.
No project/local synthetic generator is used.

This keeps the v6 3x3x3 manifold lattice parameterization and changes one thing:
inter-cell transport. Instead of predecessor mean/max, each earlier cell can send a
message only in proportion to a differentiable geometric affinity. The affinity is
high when the two learned operating points are close AND each point approximately
satisfies the other cell's implicit manifold equation.

Three parameter-matched models are trained on identical banks/batches/seeds:
  A) ordinary MLP
  B) v6 mean/max manifold lattice
  C) v7 intersection-routed manifold lattice

The output readout is intentionally left unchanged so the routing rule is the main
architectural variable under test.
"""
import copy
import math
import random

import numpy as np
import torch

import cpu_v6_manifold_lattice_deepmind as v6

OFFICIAL_DATA_CONTRACT = "google-deepmind/mathematics_dataset polynomial_roots only"
SEEDS = (11, 22, 33)
STEPS = 5000
CHECKPOINTS = {1, 100, 250, 500, 1000, 2000, 3000, 4000, 5000}
TEMPERATURE = 0.55
CROSS_RESIDUAL_WEIGHT = 0.50
DEVICE = torch.device("cpu")


class IntersectionLattice3D(v6.ManifoldLattice3D):
    """Same 27 cells/parameters as v6, but messages are routed by manifold affinity."""

    @staticmethod
    def _implicit_residual(point, family):
        x, y, z = point[..., 0], point[..., 1], point[..., 2]
        if family == 0:  # sphere: x^2+y^2+z^2=1
            return x*x + y*y + z*z - 1.0
        if family == 1:  # cylinder: x^2+y^2=1, z is free
            return x*x + y*y - 1.0
        if family == 2:  # saddle: z=x*y
            return z - x*y
        # wave surface: z=tanh(sin(pi*x)+cos(pi*y))
        return z - torch.tanh(torch.sin(math.pi*x) + torch.cos(math.pi*y))

    def _points_and_gates(self):
        points = []
        families = []
        idx = 0
        for z in range(self.SIDE):
            for y in range(self.SIDE):
                for xx in range(self.SIDE):
                    family = (xx + 2*y + 3*z) % 4
                    points.append(self._point(self.theta[idx], self.phi[idx], family))
                    families.append(family)
                    idx += 1
        p = torch.stack(points, dim=0)  # [27, width, 3]
        n = p.shape[0]

        # eval_by_family[f, j, w] = residual of point(j,w) in manifold f.
        eval_by_family = torch.stack(
            [self._implicit_residual(p, family) for family in range(4)], dim=0
        )
        fam = torch.tensor(families, dtype=torch.long, device=p.device)
        # cross_eval[i,j,w] = residual of source point j in target i's manifold.
        cross_eval = eval_by_family[fam]
        mutual_residual = cross_eval.square() + cross_eval.transpose(0, 1).square()

        delta = p[:, None, :, :] - p[None, :, :, :]
        distance_sq = delta.square().sum(dim=-1)
        energy = distance_sq + CROSS_RESIDUAL_WEIGHT * mutual_residual
        gate = torch.exp(-energy / TEMPERATURE)

        # Feed-forward only: source j may route to target i iff j < i.
        causal = torch.tril(
            torch.ones((n, n), dtype=torch.bool, device=p.device), diagonal=-1
        )
        gate = gate * causal[:, :, None]
        return p, gate, causal

    def forward(self, x, degree):
        del degree
        base = torch.tanh(self.input(x))
        points, all_gates, _ = self._points_and_gates()
        states = []

        for idx in range(27):
            point = points[idx]
            if idx == 0:
                routed = torch.zeros_like(base)
                intersection_density = torch.zeros(self.WIDTH, device=x.device, dtype=x.dtype)
            else:
                # No mean/max transport here. Geometry decides every source weight.
                gates = all_gates[idx, :idx, :]                 # [sources, width]
                previous = torch.stack(states, dim=0)           # [sources, batch, width]
                numerator = (gates[:, None, :] * previous).sum(dim=0)
                gate_sum = gates.sum(dim=0)
                routed = numerator / (1.0 + gate_sum[None, :])
                intersection_density = gate_sum / (1.0 + gate_sum)

            h = torch.tanh(
                base * point[:, 0]
                + routed * point[:, 1]
                + intersection_density[None, :] * point[:, 2]
                + self.bias[idx]
            )
            states.append(h)

        # Keep v6 readout pooling unchanged to isolate the routing intervention.
        stacked = torch.stack(states, dim=1)
        pooled = torch.cat(
            (stacked.mean(dim=1), stacked.max(dim=1).values, states[-1]), dim=1
        )
        out = self.readout(pooled)
        return out[:, :v6.ROOT_SLOTS], out[:, v6.ROOT_SLOTS:]

    @torch.no_grad()
    def routing_snapshot(self):
        _, gate, causal = self._points_and_gates()
        channel_values = gate[causal]
        edge_strength = gate.mean(dim=-1)[causal]
        return {
            "mean_channel_gate": float(channel_values.mean()),
            "median_channel_gate": float(channel_values.median()),
            "edges_gt_0.25": int((edge_strength > 0.25).sum()),
            "edges_gt_0.50": int((edge_strength > 0.50).sum()),
            "max_edge_strength": float(edge_strength.max()),
        }


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def geometry_change(model, initial_geo):
    current = model.geometry_snapshot()
    displacement = torch.linalg.vector_norm(current - initial_geo, dim=-1)
    pts = current.mean(dim=1)
    dist = torch.cdist(pts, pts)
    mask = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
    pair = dist[mask]
    return {
        "mean_point_displacement": float(displacement.mean()),
        "max_point_displacement": float(displacement.max()),
        "close_cell_pairs_lt_0.25": int((pair < 0.25).sum()),
        "median_cell_distance": float(pair.median()),
    }


def train_one(model_ctor, train, valid, seed, label):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = model_ctor().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=v6.LR, weight_decay=v6.WEIGHT_DECAY
    )
    batch_rng = torch.Generator().manual_seed(100000 + seed)

    initial_geo = model.geometry_snapshot().clone() if hasattr(model, "geometry_snapshot") else None
    initial_route = model.routing_snapshot() if hasattr(model, "routing_snapshot") else None
    curve = {}
    best_score = float("inf")
    best_state = None

    for step in range(1, STEPS + 1):
        ids = torch.randint(0, len(train.features), (v6.BATCH,), generator=batch_rng)
        optimizer.zero_grad(set_to_none=True)
        loss = v6.loss_fn(model, train.features[ids], train.roots[ids], train.degree[ids])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step in CHECKPOINTS:
            metrics = v6.evaluate(model, valid)
            curve[step] = metrics
            score = metrics["mae"] + 50.0*(1.0-metrics["count"]) + 20.0*(1.0-metrics["within1"])
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            route = model.routing_snapshot() if hasattr(model, "routing_snapshot") else None
            print(
                f"{label} seed={seed} step={step:4d} loss={float(loss.detach()):.6f} "
                f"VALID={metrics} ROUTING={route}", flush=True
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    final = v6.evaluate(model, valid)
    geo = geometry_change(model, initial_geo) if initial_geo is not None else None
    final_route = model.routing_snapshot() if hasattr(model, "routing_snapshot") else None
    return final, curve, geo, initial_route, final_route


def average(rows, model_name, key):
    return float(np.mean([row[model_name][key] for row in rows]))


def main():
    print("DEVICE=cpu")
    print("DATA_CONTRACT", OFFICIAL_DATA_CONTRACT)
    dm = v6.prepare_official_deepmind()
    print("Building deterministic official DeepMind banks...", flush=True)
    train = v6.build_bank(dm, v6.TRAIN_COUNT, 7301)
    valid = v6.build_bank(dm, v6.VALID_COUNT, 8301)
    print("train degree distribution", {d:int((train.degree==d).sum()) for d in range(2,6)})
    print("valid degree distribution", {d:int((valid.degree==d).sum()) for d in range(2,6)})

    p_mlp = parameter_count(v6.OrdinaryMLP())
    p_meanmax = parameter_count(v6.ManifoldLattice3D())
    p_intersection = parameter_count(IntersectionLattice3D())
    print(f"PARAMS mlp={p_mlp} meanmax={p_meanmax} intersection={p_intersection}")
    if p_meanmax != p_intersection:
        raise SystemExit("routing test invalid: manifold parameter counts differ")
    if abs(p_intersection-p_mlp)/p_mlp > 0.08:
        raise SystemExit("routing test invalid: MLP parameter budget mismatch >8%")

    rows = []
    for seed in SEEDS:
        mlp, mlp_curve, _, _, _ = train_one(v6.OrdinaryMLP, train, valid, seed, "MLP")
        meanmax, mm_curve, mm_geo, _, _ = train_one(v6.ManifoldLattice3D, train, valid, seed, "MEANMAX")
        inter, int_curve, int_geo, route0, route1 = train_one(IntersectionLattice3D, train, valid, seed, "INTERSECTION")
        row = {"seed":seed, "mlp":mlp, "meanmax":meanmax, "intersection":inter}
        rows.append(row)
        print(
            f"SEED_SUMMARY seed={seed} MLP={mlp} MEANMAX={meanmax} "
            f"INTERSECTION={inter} MEANMAX_GEO={mm_geo} INTERSECTION_GEO={int_geo} "
            f"ROUTE_INITIAL={route0} ROUTE_FINAL={route1}", flush=True
        )
        print("LATE_CURVE", seed, {
            step: {
                "mlp_mae": mlp_curve[step]["mae"],
                "meanmax_mae": mm_curve[step]["mae"],
                "intersection_mae": int_curve[step]["mae"],
                "mlp_count": mlp_curve[step]["count"],
                "meanmax_count": mm_curve[step]["count"],
                "intersection_count": int_curve[step]["count"],
                "mlp_within1": mlp_curve[step]["within1"],
                "meanmax_within1": mm_curve[step]["within1"],
                "intersection_within1": int_curve[step]["within1"],
            }
            for step in sorted(CHECKPOINTS)
        }, flush=True)

    keys = ("count", "mae", "within1", "merge")
    summary = {
        name: {key:average(rows, name, key) for key in keys}
        for name in ("mlp", "meanmax", "intersection")
    }
    summary["delta_vs_meanmax"] = {
        "count": summary["intersection"]["count"] - summary["meanmax"]["count"],
        "mae": summary["intersection"]["mae"] - summary["meanmax"]["mae"],
        "within1": summary["intersection"]["within1"] - summary["meanmax"]["within1"],
        "merge": summary["intersection"]["merge"] - summary["meanmax"]["merge"],
    }
    summary["delta_vs_mlp"] = {
        "count": summary["intersection"]["count"] - summary["mlp"]["count"],
        "mae": summary["intersection"]["mae"] - summary["mlp"]["mae"],
        "within1": summary["intersection"]["within1"] - summary["mlp"]["within1"],
        "merge": summary["intersection"]["merge"] - summary["mlp"]["merge"],
    }

    wins_meanmax = sum(
        row["intersection"]["mae"] < row["meanmax"]["mae"]
        and row["intersection"]["count"] >= row["meanmax"]["count"]
        for row in rows
    )
    wins_mlp = sum(
        row["intersection"]["mae"] < row["mlp"]["mae"]
        and row["intersection"]["count"] >= row["mlp"]["count"]
        for row in rows
    )

    print("FINAL_SUMMARY", summary, flush=True)
    print(f"INTERSECTION_WINS_VS_MEANMAX={wins_meanmax}/{len(SEEDS)}", flush=True)
    print(f"INTERSECTION_WINS_VS_MLP={wins_mlp}/{len(SEEDS)}", flush=True)

    improved_routing = (
        wins_meanmax >= 2
        and summary["delta_vs_meanmax"]["mae"] <= -1.0
        and summary["delta_vs_meanmax"]["within1"] >= 0.0
    )
    if improved_routing:
        print("V7_INTERSECTION_ROUTING_DEEPMIND_IMPROVED_ROUTING")
    else:
        print("V7_INTERSECTION_ROUTING_DEEPMIND_NO_IMPROVEMENT")


if __name__ == "__main__":
    main()
