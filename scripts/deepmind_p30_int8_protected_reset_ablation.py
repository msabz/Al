#!/usr/bin/env python3
"""Controlled 3-way ablation for the adopted P30 INT8-QAT RSNN.

Question
--------
Does periodically reinitializing weak *existing* P30 synapses help the RSNN,
and does protecting repeatedly useful synapses improve that reset mechanism?

Variants (run together under one paired protocol)
-------------------------------------------------
1) P30_INT8_QAT_BASELINE
   Exact adopted P30 INT8-QAT recipe, fixed T=25.
2) P30_INT8_QAT_SELECTIVE_RESET
   Same topology and active-weight budget. At fixed post-pruning checkpoints,
   reset the weakest 5% of active weights per matrix using training-gradient
   utility only. Masks never change.
3) P30_INT8_QAT_PROTECTED_RESET
   Same selective reset, but active temporary synapses that remain in the top
   30% utility for two consecutive reset checkpoints become protected and can
   no longer be reset.

Fairness / scope
----------------
- Exact same pinned Google DeepMind mathematics_dataset linear_2d banks.
- 5 paired seeds, same initialization, batches, optimizer, pruning and QAT.
- Exactly 18,816 active weights (P30) for all three variants throughout.
- No official/stress/ill metric is used to choose/reset/protect a connection.
- Utility is training-only EMA(|weight * gradient|).
- Reset does not grow/prune/rewire topology; it reinitializes weight values only.
- Weight-only symmetric per-output-channel INT8 export; states stay FP32.
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
import deepmind_rsnn_pruning_sweep as sweep
import deepmind_p30_int8_ablation as qmod

DEVICE = torch.device("cpu")
FINAL_SPARSITY = 0.30
ACTIVE_EXPECTED = 18816
BASELINE = "P30_INT8_QAT_BASELINE"
RESET = "P30_INT8_QAT_SELECTIVE_RESET"
PROTECTED = "P30_INT8_QAT_PROTECTED_RESET"
LABELS = (BASELINE, RESET, PROTECTED)
RESET_EPOCHS = (160, 180, 200, 220, 240)
RESET_FRACTION = 0.05
UTILITY_BETA = 0.95
PROTECT_TOP_FRACTION = 0.30
PROTECT_STREAK = 2
RESET_INIT_SCALE = 0.01
METRICS = qmod.METRICS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def active_count(model: arch.PrunedBrainSpikeNet) -> int:
    return sum(int((m > 0.5).sum().item()) for m in (model.M_in, model.M_rec, model.M_out))


class ResetQATP30RSNN(qmod.QATP30RSNN):
    def __init__(self, *, protection: bool) -> None:
        super().__init__()
        self.protection_enabled = bool(protection)
        self.reset_events: List[Dict[str, object]] = []
        for n, w in (("in", self.W_in), ("rec", self.W_rec), ("out", self.W_out)):
            self.register_buffer(f"utility_{n}", torch.zeros_like(w))
            self.register_buffer(f"streak_{n}", torch.zeros_like(w, dtype=torch.int16))
            self.register_buffer(f"protected_{n}", torch.zeros_like(w, dtype=torch.bool))

    def _triples(self):
        return (
            ("in", self.W_in, self.M_in),
            ("rec", self.W_rec, self.M_rec),
            ("out", self.W_out, self.M_out),
        )

    @torch.no_grad()
    def update_utility_(self) -> None:
        for n, w, mask in self._triples():
            if w.grad is None:
                continue
            active = mask > 0.5
            score = (w.detach() * w.grad.detach()).abs()
            utility = getattr(self, f"utility_{n}")
            utility.mul_(UTILITY_BETA)
            utility.add_((1.0 - UTILITY_BETA) * score * active)
            utility.mul_(active)

    @torch.no_grad()
    def _zero_optimizer_positions_(self, optimizer: torch.optim.Optimizer, w: nn.Parameter, changed: torch.Tensor) -> None:
        state = optimizer.state.get(w, {})
        for value in state.values():
            if torch.is_tensor(value) and value.shape == w.shape:
                value[changed] = 0

    @torch.no_grad()
    def _promote_(self, n: str, mask: torch.Tensor) -> Dict[str, int]:
        protected = getattr(self, f"protected_{n}")
        streak = getattr(self, f"streak_{n}")
        utility = getattr(self, f"utility_{n}")
        active = mask > 0.5
        if not self.protection_enabled:
            protected.zero_()
            streak.zero_()
            return {"protected_before": 0, "newly_protected": 0, "protected_after": 0}

        before = int(protected.sum().item())
        vals = utility[active]
        if vals.numel() == 0:
            return {"protected_before": before, "newly_protected": 0, "protected_after": before}
        k = max(1, int(round(vals.numel() * PROTECT_TOP_FRACTION)))
        threshold = torch.topk(vals, k=k, largest=True).values.min()
        temporary = active & (~protected)
        strong = temporary & (utility >= threshold)
        weak = temporary & (~strong)
        streak[strong] += 1
        streak[weak] = 0
        newly_mask = temporary & (streak >= PROTECT_STREAK)
        protected |= newly_mask
        after = int(protected.sum().item())
        return {"protected_before": before, "newly_protected": after - before, "protected_after": after}

    @torch.no_grad()
    def reset_weak_(self, optimizer: torch.optim.Optimizer, *, seed: int, epoch: int) -> Dict[str, object]:
        event: Dict[str, object] = {"epoch": epoch, "protection_enabled": self.protection_enabled, "matrices": {}}
        active_before_total = active_count(self)
        for matrix_id, (n, w, mask) in enumerate(self._triples()):
            promotion = self._promote_(n, mask)
            utility = getattr(self, f"utility_{n}")
            protected = getattr(self, f"protected_{n}")
            streak = getattr(self, f"streak_{n}")
            active = mask > 0.5
            active_n = int(active.sum().item())
            eligible = active & (~protected if self.protection_enabled else torch.ones_like(active, dtype=torch.bool))
            eligible_idx = torch.nonzero(eligible.view(-1), as_tuple=False).flatten()
            target = max(1, int(round(active_n * RESET_FRACTION)))
            nreset = min(target, int(eligible_idx.numel()))
            if nreset <= 0:
                event["matrices"][n] = {**promotion, "active": active_n, "reset": 0}
                continue

            flat_u = utility.view(-1)
            local = torch.topk(flat_u[eligible_idx], k=nreset, largest=False, sorted=False).indices
            ids = eligible_idx[local]
            changed = torch.zeros_like(mask, dtype=torch.bool).view(-1)
            changed[ids] = True
            changed = changed.view_as(mask)
            if self.protection_enabled and bool((changed & protected).any()):
                raise AssertionError(f"protected synapse selected for reset matrix={n}")

            active_std = float(w.detach()[active].std().item()) if bool(active.any()) else 0.01
            init_std = max(active_std * RESET_INIT_SCALE, 1e-5)
            gen = torch.Generator(device=w.device)
            gen.manual_seed(seed * 100000 + epoch * 100 + matrix_id)
            new_values = torch.randn(nreset, generator=gen, device=w.device, dtype=w.dtype) * init_std
            w.view(-1)[ids] = new_values
            utility.view(-1)[ids] = 0.0
            streak.view(-1)[ids] = 0
            self._zero_optimizer_positions_(optimizer, w, changed)

            event["matrices"][n] = {
                **promotion,
                "active": active_n,
                "eligible": int(eligible.sum().item()),
                "reset": nreset,
                "reset_fraction_active": nreset / max(active_n, 1),
                "reset_init_std": init_std,
            }

        self.apply_masks_()
        active_after_total = active_count(self)
        if active_after_total != active_before_total:
            raise AssertionError(f"reset changed active topology {active_before_total}->{active_after_total}")
        event["active_total_before"] = active_before_total
        event["active_total_after"] = active_after_total
        event["protected_total"] = sum(int(getattr(self, f"protected_{n}").sum().item()) for n in ("in", "rec", "out"))
        self.reset_events.append(event)
        return event

    @torch.no_grad()
    def reset_report(self) -> Dict[str, object]:
        mats: Dict[str, object] = {}
        total_protected = 0
        for n, _, mask in self._triples():
            protected = int(getattr(self, f"protected_{n}").sum().item())
            total_protected += protected
            mats[n] = {"active": int((mask > 0.5).sum().item()), "protected": protected}
        return {
            "protection_enabled": self.protection_enabled,
            "active_total": active_count(self),
            "protected_total": total_protected,
            "protected_fraction_of_active": total_protected / max(active_count(self), 1),
            "events": self.reset_events,
            "matrices": mats,
        }


def train_reset_variant(
    model: ResetQATP30RSNN,
    train: dm.Bank,
    monitor: dm.Bank,
    *, epochs: int, batch_size: int, seed: int, label: str,
) -> Tuple[ResetQATP30RSNN, Dict[str, object]]:
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arch.LR, weight_decay=arch.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=arch.ETA_MIN)
    criterion = nn.SmoothL1Loss(beta=1.0)
    checkpoints = {1, 50, 100, 150, qmod.QAT_START_EPOCH, 160, 180, 200, 220, 240, 250, epochs}
    started = time.perf_counter()
    history: List[Dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        target = sweep.pruning_target(epoch, FINAL_SPARSITY)
        if target is not None:
            report = model.prune_synapses(target, optimizer)
            print(f"PRUNE seed={seed} label={label} epoch={epoch} target={target:.4f} actual={json.dumps(report, sort_keys=True)}", flush=True)

        model.quant_enabled = epoch >= qmod.QAT_START_EPOCH
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
            if epoch >= qmod.QAT_START_EPOCH:
                model.update_utility_()
            model.zero_pruned_gradients_()
            optimizer.step()
            model.apply_masks_()
            running += float(loss.detach()) * len(ids)
            seen += len(ids)
        scheduler.step()

        if epoch in RESET_EPOCHS:
            before_masks = {n: m.detach().clone() for n, m in (("in", model.M_in), ("rec", model.M_rec), ("out", model.M_out))}
            event = model.reset_weak_(optimizer, seed=seed, epoch=epoch)
            for n, m in (("in", model.M_in), ("rec", model.M_rec), ("out", model.M_out)):
                if not torch.equal(before_masks[n], m):
                    raise AssertionError(f"topology changed during reset seed={seed} label={label} matrix={n}")
            print(f"SELECTIVE_RESET seed={seed} label={label} {json.dumps(event, sort_keys=True)}", flush=True)

        if epoch in checkpoints:
            row = {
                "epoch": epoch,
                "loss": running / max(seen, 1),
                "quant_enabled": model.quant_enabled,
                "official_monitor": dm.evaluate_raw(model, monitor),
                "sparsity": model.sparsity_report(),
                "reset_report": model.reset_report(),
            }
            history.append(row)
            print(f"CHECKPOINT seed={seed} label={label} epoch={epoch} loss={row['loss']:.8f}", flush=True)

    model.quant_enabled = True
    if active_count(model) != ACTIVE_EXPECTED:
        raise AssertionError(f"active contract failed seed={seed} label={label}: {active_count(model)}")
    sparsity = model.sparsity_report()
    for key in ("W_in", "W_rec", "W_out", "overall"):
        if abs(float(sparsity[key]) - FINAL_SPARSITY) > 0.002:
            raise AssertionError(f"sparsity mismatch seed={seed} label={label} {key}={sparsity[key]}")
    return model, {
        "seed": seed,
        "label": label,
        "training": "selective_reset_qat" if label == RESET else "protected_core_selective_reset_qat",
        "params": arch.count_parameters(model),
        "active_weights": ACTIVE_EXPECTED,
        "train_seconds": time.perf_counter() - started,
        "sparsity": sparsity,
        "history": history,
        "reset_report": model.reset_report(),
    }


def attach_eval(result: Dict[str, object], model: nn.Module, banks: Dict[str, dm.Bank]) -> None:
    result["evaluations"] = {name: dm.evaluate_raw(model, bank) for name, bank in banks.items()}
    print(f"SEED_FINAL seed={result['seed']} label={result['label']} {json.dumps(result['evaluations'], sort_keys=True)}", flush=True)


def aggregate(runs: List[Dict[str, object]], bank_names: Sequence[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    seeds = sorted({int(r["seed"]) for r in runs})
    for label in LABELS:
        rows = [r for r in runs if r["label"] == label]
        if len(rows) != len(seeds):
            raise AssertionError(f"missing rows label={label}")
        entry: Dict[str, object] = {
            "params": int(rows[0]["params"]),
            "active_weights": int(rows[0]["active_weights"]),
            "train_seconds": {"mean": mean(float(r["train_seconds"]) for r in rows), "std": pstdev(float(r["train_seconds"]) for r in rows)},
            "banks": {},
        }
        for bank in bank_names:
            entry["banks"][bank] = {}
            for metric in METRICS:
                vals = [float(r["evaluations"][bank][metric]) for r in rows]
                entry["banks"][bank][metric] = {"mean": mean(vals), "std": pstdev(vals), "min": min(vals), "max": max(vals)}
        out[label] = entry
    return out


def paired(runs: List[Dict[str, object]], reference: str, candidate: str, bank_names: Sequence[str]) -> Dict[str, object]:
    refs = {int(r["seed"]): r for r in runs if r["label"] == reference}
    cands = {int(r["seed"]): r for r in runs if r["label"] == candidate}
    if set(refs) != set(cands):
        raise AssertionError(f"seed mismatch {reference} vs {candidate}")
    out: Dict[str, object] = {}
    for bank in bank_names:
        mae_pct: List[float] = []
        strict_pp: List[float] = []
        within_pp: List[float] = []
        residual_pct: List[float] = []
        wins = 0
        for seed in sorted(refs):
            a = refs[seed]["evaluations"][bank]
            b = cands[seed]["evaluations"][bank]
            mae_pct.append(100.0 * (float(b["mae"]) / float(a["mae"]) - 1.0))
            strict_pp.append(100.0 * (float(b["strict_1_0"]) - float(a["strict_1_0"])))
            within_pp.append(100.0 * (float(b["within_5_0"]) - float(a["within_5_0"])))
            residual_pct.append(100.0 * (float(b["equation_residual_mean"]) / float(a["equation_residual_mean"]) - 1.0))
            wins += int(float(b["mae"]) < float(a["mae"]))
        out[bank] = {
            "mae_change_pct_mean": mean(mae_pct),
            "mae_change_pct_std": pstdev(mae_pct),
            "strict1_change_pp_mean": mean(strict_pp),
            "strict1_change_pp_std": pstdev(strict_pp),
            "within5_change_pp_mean": mean(within_pp),
            "residual_change_pct_mean": mean(residual_pct),
            "mae_wins": wins,
            "mae_losses": len(refs) - wins,
        }
    return out


def preflight() -> None:
    set_seed(1234)
    m = ResetQATP30RSNN(protection=False)
    opt = torch.optim.AdamW(m.parameters(), lr=arch.LR, weight_decay=arch.WEIGHT_DECAY)
    m.prune_synapses(FINAL_SPARSITY, opt)
    if active_count(m) != ACTIVE_EXPECTED:
        raise AssertionError(active_count(m))
    masks = {n: x.detach().clone() for n, x in (("in", m.M_in), ("rec", m.M_rec), ("out", m.M_out))}
    m.quant_enabled = True
    x = torch.randn(32, arch.IN_DIM)
    loss = m(x).square().mean()
    loss.backward()
    m.update_utility_()
    event = m.reset_weak_(opt, seed=1234, epoch=160)
    if event["active_total_after"] != ACTIVE_EXPECTED:
        raise AssertionError(event)
    for n, xmask in (("in", m.M_in), ("rec", m.M_rec), ("out", m.M_out)):
        if not torch.equal(masks[n], xmask):
            raise AssertionError(f"mask changed in reset preflight {n}")

    set_seed(1234)
    p = ResetQATP30RSNN(protection=True)
    p.prune_synapses(FINAL_SPARSITY, torch.optim.AdamW(p.parameters(), lr=arch.LR, weight_decay=arch.WEIGHT_DECAY))
    p.quant_enabled = True
    optp = torch.optim.AdamW(p.parameters(), lr=arch.LR, weight_decay=arch.WEIGHT_DECAY)
    for epoch in (160, 180):
        optp.zero_grad(set_to_none=True)
        p(torch.randn(32, arch.IN_DIM)).square().mean().backward()
        p.update_utility_()
        ev = p.reset_weak_(optp, seed=1234, epoch=epoch)
    if ev["active_total_after"] != ACTIVE_EXPECTED:
        raise AssertionError(ev)
    if int(p.reset_report()["protected_total"]) <= 0:
        raise AssertionError("protected core did not form in preflight")
    exported = qmod.Int8WeightRSNN(p)
    if int(exported.storage_report()["logical_active_weights"]) != ACTIVE_EXPECTED:
        raise AssertionError(exported.storage_report())
    print(f"PROTECTED_RESET_PREFLIGHT_PASS reset={json.dumps(event, sort_keys=True)} protected_total={p.reset_report()['protected_total']}", flush=True)


def make_summary(agg: Dict[str, object], comps: Dict[str, object], runs: List[Dict[str, object]], storage: Dict[str, object]) -> str:
    official = "OFFICIAL_INTERPOLATE"
    lines = [
        "P30 INT8-QAT PROTECTED CORE + SELECTIVE RESET — DEEPMIND linear_2d",
        "",
        "Three variants in one paired run: baseline / selective reset / protected-core reset.",
        "Topology fixed at exactly 18,816 active weights for every variant; reset changes values only.",
        "Reset/protection decisions use training gradients only; official/stress/ill are final evaluation only.",
        f"reset_epochs={RESET_EPOCHS} reset_fraction={RESET_FRACTION} utility_beta={UTILITY_BETA} protect_top={PROTECT_TOP_FRACTION} protect_streak={PROTECT_STREAK}",
        "",
        "Storage payload (same for all final variants):",
        json.dumps(storage, sort_keys=True),
        "",
    ]
    for label in LABELS:
        b = agg[label]["banks"][official]
        lines.append(f"{label}: MAE={b['mae']['mean']:.8f} RMSE={b['rmse']['mean']:.8f} strict1={100*b['strict_1_0']['mean']:.4f}% within5={100*b['within_5_0']['mean']:.4f}% train_s={agg[label]['train_seconds']['mean']:.3f}")
    lines.append("")
    for name, banks in comps.items():
        lines.append(name)
        for bank, row in banks.items():
            lines.append(f"  {bank}: MAE_delta={row['mae_change_pct_mean']:+.4f}% strict1_delta={row['strict1_change_pp_mean']:+.4f}pp within5_delta={row['within5_change_pp_mean']:+.4f}pp residual_delta={row['residual_change_pct_mean']:+.4f}% wins={row['mae_wins']}/{row['mae_wins']+row['mae_losses']}")
    lines.append("")
    for r in runs:
        if r["label"] in (RESET, PROTECTED):
            rr = r["reset_report"]
            lines.append(f"seed={r['seed']} label={r['label']} protected={rr['protected_total']} protected_fraction={rr['protected_fraction_of_active']:.6f} events={len(rr['events'])}")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--train-size", type=int, default=8192)
    p.add_argument("--eval-size", type=int, default=2048)
    p.add_argument("--ill-pool-size", type=int, default=4096)
    p.add_argument("--ill-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seeds", default="11,22,33,44,55")
    p.add_argument("--output-dir", default="deepmind-p30-int8-protected-reset-output")
    args = p.parse_args()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    if args.preflight:
        preflight()
        return
    if args.epochs < 300:
        raise ValueError("full controlled test requires 300 epochs")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) != 5:
        raise ValueError("exact 5-seed contract required")

    train, banks, manifest = qmod.build_banks(args)
    params = arch.count_parameters(arch.PrunedBrainSpikeNet())
    if params != 26880 or ACTIVE_EXPECTED != 18816:
        raise AssertionError((params, ACTIVE_EXPECTED))
    print("PROTECTED_RESET_FAIRNESS_CONTRACT same DeepMind banks/seeds/init/batches/optimizer/P30/QAT; only reset/protection differs", flush=True)
    print("PROTECTED_RESET_TOPOLOGY_CONTRACT masks fixed after P30; exactly 18816 active weights; no growth/rewire", flush=True)
    print("DATA_CONTRACT pinned google-deepmind/mathematics_dataset linear_2d; project synthetic data=0", flush=True)

    runs: List[Dict[str, object]] = []
    first_storage: Dict[str, object] | None = None
    for seed in seeds:
        set_seed(seed)
        baseline = qmod.QATP30RSNN()
        init = qmod.clone_state(baseline)
        set_seed(seed)
        reset = ResetQATP30RSNN(protection=False)
        qmod.assert_same_initialization(init, qmod.clone_state(reset), f"seed={seed} reset")
        set_seed(seed)
        protected = ResetQATP30RSNN(protection=True)
        qmod.assert_same_initialization(init, qmod.clone_state(protected), f"seed={seed} protected")

        baseline, br = qmod.train_qat_variant(baseline, train, banks["OFFICIAL_INTERPOLATE"], epochs=args.epochs, batch_size=args.batch_size, seed=seed)
        br["label"] = BASELINE
        be = qmod.Int8WeightRSNN(baseline).to(DEVICE)
        br["storage"] = be.storage_report()
        attach_eval(br, be, banks)
        runs.append(br)
        if first_storage is None:
            first_storage = be.storage_report()

        reset, rr = train_reset_variant(reset, train, banks["OFFICIAL_INTERPOLATE"], epochs=args.epochs, batch_size=args.batch_size, seed=seed, label=RESET)
        re = qmod.Int8WeightRSNN(reset).to(DEVICE)
        rr["storage"] = re.storage_report()
        attach_eval(rr, re, banks)
        runs.append(rr)

        protected, pr = train_reset_variant(protected, train, banks["OFFICIAL_INTERPOLATE"], epochs=args.epochs, batch_size=args.batch_size, seed=seed, label=PROTECTED)
        pe = qmod.Int8WeightRSNN(protected).to(DEVICE)
        pr["storage"] = pe.storage_report()
        attach_eval(pr, pe, banks)
        runs.append(pr)
        print(f"SEED_COMPLETE seed={seed}", flush=True)

    bank_names = list(banks.keys())
    agg = aggregate(runs, bank_names)
    comps = {
        f"{RESET}_vs_{BASELINE}": paired(runs, BASELINE, RESET, bank_names),
        f"{PROTECTED}_vs_{BASELINE}": paired(runs, BASELINE, PROTECTED, bank_names),
        f"{PROTECTED}_vs_{RESET}": paired(runs, RESET, PROTECTED, bank_names),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "P30_INT8_QAT_PROTECTED_CORE_SELECTIVE_RESET",
        "device": str(DEVICE),
        "seeds": seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "reset_epochs": RESET_EPOCHS,
        "reset_fraction": RESET_FRACTION,
        "utility_beta": UTILITY_BETA,
        "protect_top_fraction": PROTECT_TOP_FRACTION,
        "protect_streak": PROTECT_STREAK,
        "reset_init_scale": RESET_INIT_SCALE,
        "runs": runs,
        "aggregate": agg,
        "paired": comps,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    manifest["protected_reset"] = {
        "uses_test_labels_for_decisions": False,
        "active_weight_budget": ACTIVE_EXPECTED,
        "topology_changes_after_p30": False,
        "reset_changes_weight_values_only": True,
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    summary = make_summary(agg, comps, runs, first_storage or {})
    (out / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary, flush=True)
    print("PROTECTED_RESET_EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
