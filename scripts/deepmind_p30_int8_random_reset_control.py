#!/usr/bin/env python3
"""Four-way control: baseline vs random reset vs selective reset vs protected reset.

Purpose
-------
The existing protected-reset ablation can show whether resetting weak synapses helps,
but without a random-reset control it cannot tell whether any gain comes from the
utility ranking or merely from periodic stochastic rejuvenation. This experiment
adds that missing control under the exact same DeepMind/P30/INT8-QAT protocol.

Variants
--------
1) P30_INT8_QAT_BASELINE
2) P30_INT8_QAT_RANDOM_RESET       -- reset 5% random active weights per matrix
3) P30_INT8_QAT_SELECTIVE_RESET    -- reset 5% lowest-utility active weights
4) P30_INT8_QAT_PROTECTED_RESET    -- selective reset + persistent protected core

All reset variants keep masks and the 18,816-active-weight P30 topology unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence

import torch

import pruned_snn_vs_mlp as arch
import deepmind_p30_int8_ablation as qmod
import deepmind_p30_int8_protected_reset_ablation as pr

DEVICE = torch.device("cpu")
BASELINE = pr.BASELINE
RANDOM = "P30_INT8_QAT_RANDOM_RESET"
RESET = pr.RESET
PROTECTED = pr.PROTECTED
LABELS = (BASELINE, RANDOM, RESET, PROTECTED)
METRICS = qmod.METRICS


class RandomResetQATP30RSNN(pr.ResetQATP30RSNN):
    def __init__(self) -> None:
        super().__init__(protection=False)

    @torch.no_grad()
    def reset_weak_(self, optimizer: torch.optim.Optimizer, *, seed: int, epoch: int) -> Dict[str, object]:
        event: Dict[str, object] = {"epoch": epoch, "mode": "random_reset_control", "matrices": {}}
        active_before_total = pr.active_count(self)
        for matrix_id, (n, w, mask) in enumerate(self._triples()):
            active = mask > 0.5
            active_idx = torch.nonzero(active.view(-1), as_tuple=False).flatten()
            active_n = int(active_idx.numel())
            nreset = min(max(1, int(round(active_n * pr.RESET_FRACTION))), active_n)

            gen = torch.Generator(device=w.device)
            gen.manual_seed(seed * 100000 + epoch * 100 + matrix_id + 700000001)
            perm = torch.randperm(active_n, generator=gen, device=w.device)
            ids = active_idx[perm[:nreset]]

            active_std = float(w.detach()[active].std().item()) if bool(active.any()) else 0.01
            init_std = max(active_std * pr.RESET_INIT_SCALE, 1e-5)
            value_gen = torch.Generator(device=w.device)
            value_gen.manual_seed(seed * 100000 + epoch * 100 + matrix_id + 800000003)
            new_values = torch.randn(nreset, generator=value_gen, device=w.device, dtype=w.dtype) * init_std
            w.view(-1)[ids] = new_values

            getattr(self, f"utility_{n}").view(-1)[ids] = 0.0
            getattr(self, f"streak_{n}").view(-1)[ids] = 0
            changed = torch.zeros_like(mask, dtype=torch.bool).view(-1)
            changed[ids] = True
            self._zero_optimizer_positions_(optimizer, w, changed.view_as(mask))

            event["matrices"][n] = {
                "active": active_n,
                "reset": nreset,
                "reset_fraction_active": nreset / max(active_n, 1),
                "reset_init_std": init_std,
            }

        self.apply_masks_()
        after = pr.active_count(self)
        if after != active_before_total:
            raise AssertionError(f"random reset changed topology {active_before_total}->{after}")
        event["active_total_before"] = active_before_total
        event["active_total_after"] = after
        self.reset_events.append(event)
        return event


def attach_eval(result: Dict[str, object], model: torch.nn.Module, banks) -> None:
    result["evaluations"] = {name: qmod.dm.evaluate_raw(model, bank) for name, bank in banks.items()}
    print(f"SEED_FINAL seed={result['seed']} label={result['label']} {json.dumps(result['evaluations'], sort_keys=True)}", flush=True)


def aggregate(runs: List[Dict[str, object]], bank_names: Sequence[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for label in LABELS:
        rows = [r for r in runs if r["label"] == label]
        entry = {
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
    out: Dict[str, object] = {}
    for bank in bank_names:
        mae, strict, within, residual = [], [], [], []
        wins = 0
        for seed in sorted(refs):
            a = refs[seed]["evaluations"][bank]
            b = cands[seed]["evaluations"][bank]
            mae.append(100.0 * (float(b["mae"]) / float(a["mae"]) - 1.0))
            strict.append(100.0 * (float(b["strict_1_0"]) - float(a["strict_1_0"])))
            within.append(100.0 * (float(b["within_5_0"]) - float(a["within_5_0"])))
            residual.append(100.0 * (float(b["equation_residual_mean"]) / float(a["equation_residual_mean"]) - 1.0))
            wins += int(float(b["mae"]) < float(a["mae"]))
        out[bank] = {
            "mae_change_pct_mean": mean(mae), "mae_change_pct_std": pstdev(mae),
            "strict1_change_pp_mean": mean(strict), "within5_change_pp_mean": mean(within),
            "residual_change_pct_mean": mean(residual), "mae_wins": wins, "mae_losses": len(refs)-wins,
        }
    return out


def preflight() -> None:
    pr.set_seed(1234)
    m = RandomResetQATP30RSNN()
    opt = torch.optim.AdamW(m.parameters(), lr=arch.LR, weight_decay=arch.WEIGHT_DECAY)
    m.prune_synapses(pr.FINAL_SPARSITY, opt)
    before_masks = [m.M_in.clone(), m.M_rec.clone(), m.M_out.clone()]
    before = pr.active_count(m)
    event = m.reset_weak_(opt, seed=1234, epoch=160)
    after = pr.active_count(m)
    if before != pr.ACTIVE_EXPECTED or after != pr.ACTIVE_EXPECTED:
        raise AssertionError((before, after))
    if not all(torch.equal(a, b) for a, b in zip(before_masks, (m.M_in, m.M_rec, m.M_out))):
        raise AssertionError("random reset changed masks")
    if sum(int(x["reset"]) for x in event["matrices"].values()) <= 0:
        raise AssertionError("random reset did not reset positions")
    print(f"RANDOM_RESET_CONTROL_PREFLIGHT_PASS active={after} event={json.dumps(event, sort_keys=True)}", flush=True)


def make_summary(agg, comps) -> str:
    off = "OFFICIAL_INTERPOLATE"
    lines = ["P30 INT8-QAT RANDOM RESET CONTROL — DEEPMIND linear_2d", "",
             "Four-way paired control: baseline, random reset, selective reset, protected reset."]
    for label in LABELS:
        b = agg[label]["banks"][off]
        lines.append(f"{label}: MAE={b['mae']['mean']:.8f} RMSE={b['rmse']['mean']:.8f} strict1={100*b['strict_1_0']['mean']:.4f}% within5={100*b['within_5_0']['mean']:.4f}%")
    lines.append("")
    for key, banks in comps.items():
        lines.append(key)
        for bank, row in banks.items():
            lines.append(f"  {bank}: MAE_delta={row['mae_change_pct_mean']:+.4f}% strict1_delta={row['strict1_change_pp_mean']:+.4f}pp within5_delta={row['within5_change_pp_mean']:+.4f}pp residual_delta={row['residual_change_pct_mean']:+.4f}% wins={row['mae_wins']}/{row['mae_wins']+row['mae_losses']}")
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
    p.add_argument("--output-dir", default="deepmind-p30-int8-random-reset-control-output")
    args = p.parse_args()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    if args.preflight:
        preflight(); return

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    train, banks, manifest = qmod.build_banks(args)
    runs: List[Dict[str, object]] = []
    print("RANDOM_RESET_CONTROL_FAIRNESS same init/data/batches/optimizer/P30/QAT/reset schedule; reset selection rule is the controlled factor", flush=True)
    print("DATA_CONTRACT pinned google-deepmind/mathematics_dataset linear_2d; project synthetic data=0", flush=True)

    for seed in seeds:
        pr.set_seed(seed); init_model = qmod.QATP30RSNN(); init = qmod.clone_state(init_model)

        pr.set_seed(seed); base = qmod.QATP30RSNN(); qmod.assert_same_initialization(init, qmod.clone_state(base), f"seed={seed} baseline")
        pr.set_seed(seed); rnd = RandomResetQATP30RSNN(); qmod.assert_same_initialization(init, qmod.clone_state(rnd), f"seed={seed} random")
        pr.set_seed(seed); sel = pr.ResetQATP30RSNN(protection=False); qmod.assert_same_initialization(init, qmod.clone_state(sel), f"seed={seed} selective")
        pr.set_seed(seed); prot = pr.ResetQATP30RSNN(protection=True); qmod.assert_same_initialization(init, qmod.clone_state(prot), f"seed={seed} protected")

        base, br = qmod.train_qat_variant(base, train, banks["OFFICIAL_INTERPOLATE"], epochs=args.epochs, batch_size=args.batch_size, seed=seed)
        br["label"] = BASELINE
        be = qmod.Int8WeightRSNN(base).to(DEVICE); attach_eval(br, be, banks); runs.append(br)

        rnd, rr = pr.train_reset_variant(rnd, train, banks["OFFICIAL_INTERPOLATE"], epochs=args.epochs, batch_size=args.batch_size, seed=seed, label=RANDOM)
        rr["training"] = "random_reset_qat_control"
        re = qmod.Int8WeightRSNN(rnd).to(DEVICE); attach_eval(rr, re, banks); runs.append(rr)

        sel, sr = pr.train_reset_variant(sel, train, banks["OFFICIAL_INTERPOLATE"], epochs=args.epochs, batch_size=args.batch_size, seed=seed, label=RESET)
        se = qmod.Int8WeightRSNN(sel).to(DEVICE); attach_eval(sr, se, banks); runs.append(sr)

        prot, prr = pr.train_reset_variant(prot, train, banks["OFFICIAL_INTERPOLATE"], epochs=args.epochs, batch_size=args.batch_size, seed=seed, label=PROTECTED)
        pe = qmod.Int8WeightRSNN(prot).to(DEVICE); attach_eval(prr, pe, banks); runs.append(prr)
        print(f"SEED_COMPLETE seed={seed}", flush=True)

    bank_names = list(banks.keys())
    agg = aggregate(runs, bank_names)
    comps = {
        f"{RANDOM}_vs_{BASELINE}": paired(runs, BASELINE, RANDOM, bank_names),
        f"{RESET}_vs_{BASELINE}": paired(runs, BASELINE, RESET, bank_names),
        f"{PROTECTED}_vs_{BASELINE}": paired(runs, BASELINE, PROTECTED, bank_names),
        f"{RESET}_vs_{RANDOM}": paired(runs, RANDOM, RESET, bank_names),
        f"{PROTECTED}_vs_{RANDOM}": paired(runs, RANDOM, PROTECTED, bank_names),
        f"{PROTECTED}_vs_{RESET}": paired(runs, RESET, PROTECTED, bank_names),
    }

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    results = {"experiment":"P30_INT8_QAT_RANDOM_RESET_CONTROL", "device":str(DEVICE), "seeds":seeds,
               "epochs":args.epochs, "reset_epochs":pr.RESET_EPOCHS, "reset_fraction":pr.RESET_FRACTION,
               "runs":runs, "aggregate":agg, "paired":comps}
    (out/"results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    manifest["random_reset_control"] = {"active_weight_budget":pr.ACTIVE_EXPECTED, "topology_changes":False, "uses_test_for_reset":False}
    (out/"dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    summary = make_summary(agg, comps)
    (out/"SUMMARY.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary, flush=True)
    print("RANDOM_RESET_CONTROL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
