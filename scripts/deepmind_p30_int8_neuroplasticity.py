#!/usr/bin/env python3
"""Controlled neuroplasticity-inspired ablation for the adopted P30 INT8-QAT RSNN.

This is a computational analogue, not a claim of biological equivalence.

Baseline
--------
P30_INT8_QAT_BASELINE: adopted 30%-pruned weight-only INT8-QAT RSNN, fixed T=25.

Variants
--------
P30_INT8_QAT_REWIRE:
  - Same initialization/data/batches/optimizer/pruning/QAT as baseline.
  - After pruning is complete, inactive synapses receive a zero-forward/gradient-only
    proxy so we can measure task-relevant dormant gradient signal.
  - Every 20 epochs from 160..240, 5% of active synapses per matrix with the lowest
    utility EMA are removed and the same number of dormant synapses with the largest
    dormant-gradient EMA are activated. Active-count remains exactly P30.

P30_INT8_QAT_CONSOLIDATED:
  - Same rewiring rule.
  - At each rewiring event, active temporary synapses in the top 30% of utility earn
    one persistence mark; two consecutive marks promote them to consolidated.
  - Consolidated synapses are protected from later pruning. New synapses begin
    temporary and must prove utility before becoming consolidated.

Biological inspiration mapping (approximate)
--------------------------------------------
- temporary synapse      -> active but unconsolidated connection
- repeated useful use    -> EMA(|w * grad|) task-utility + persistence streak
- salience/modulation    -> loss-gradient magnitude naturally scales utility signal
- consolidation/offline  -> promotion decision at discrete rewiring checkpoints
- pruning                -> remove low-utility temporary connections
- synaptogenesis         -> activate high dormant-gradient candidates

Fairness
--------
- Same pinned DeepMind mathematics_dataset linear_2d banks, 5 seeds, 300 epochs.
- Same final 30% sparsity (18,816 active of 26,880) for all variants.
- Same weight-only INT8-QAT scheme; activations/membrane/accumulators stay FP32.
- No test-set metric is used for rewiring or consolidation.
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
import torch.nn.functional as F

import pruned_snn_vs_mlp as arch
import deepmind_pruned_snn_stress as dm
import deepmind_rsnn_pruning_sweep as sweep
import deepmind_p30_int8_ablation as qmod

DEVICE = torch.device("cpu")
BASELINE = "P30_INT8_QAT_BASELINE"
REWIRE = "P30_INT8_QAT_REWIRE"
CONSOLIDATED = "P30_INT8_QAT_CONSOLIDATED"
LABELS = (BASELINE, REWIRE, CONSOLIDATED)
FINAL_SPARSITY = 0.30
ACTIVE_EXPECTED = 18816
PLASTICITY_START = 151
REWIRE_EPOCHS = (160, 180, 200, 220, 240)
REWIRE_FRACTION = 0.05
UTILITY_BETA = 0.95
DORMANT_BETA = 0.95
CONSOLIDATE_TOP_FRACTION = 0.30
CONSOLIDATE_STREAK = 2
METRICS = qmod.METRICS


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


class PlasticQATP30RSNN(qmod.QATP30RSNN):
    def __init__(self, *, consolidation: bool) -> None:
        super().__init__()
        self.consolidation_enabled = bool(consolidation)
        self.proxy_enabled = False
        for n, w in (("in", self.W_in), ("rec", self.W_rec), ("out", self.W_out)):
            self.register_buffer(f"utility_{n}", torch.zeros_like(w))
            self.register_buffer(f"dormant_{n}", torch.zeros_like(w))
            self.register_buffer(f"streak_{n}", torch.zeros_like(w, dtype=torch.int16))
            self.register_buffer(f"consolidated_{n}", torch.zeros_like(w, dtype=torch.bool))
        self.rewire_events: List[Dict[str, object]] = []

    def _effective(self, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = weight * mask
        if self.proxy_enabled:
            # Forward contribution of inactive weights is exactly zero, but backward
            # derivative is one, yielding a dense dormant-gradient candidate signal.
            masked = masked + (weight - weight.detach()) * (1.0 - mask)
        if self.quant_enabled:
            masked = qmod.fake_quant_per_channel_ste(masked)
        return masked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_in = self._effective(self.W_in, self.M_in)
        w_rec = self._effective(self.W_rec, self.M_rec)
        w_out = self._effective(self.W_out, self.M_out)
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

    def _triples(self):
        return (
            ("in", self.W_in, self.M_in),
            ("rec", self.W_rec, self.M_rec),
            ("out", self.W_out, self.M_out),
        )

    @torch.no_grad()
    def update_plasticity_scores_(self) -> None:
        for n, w, mask in self._triples():
            if w.grad is None:
                continue
            g = w.grad.detach()
            active = mask > 0.5
            inactive = ~active
            utility = getattr(self, f"utility_{n}")
            dormant = getattr(self, f"dormant_{n}")
            utility.mul_(UTILITY_BETA)
            utility.add_((1.0 - UTILITY_BETA) * (w.detach() * g).abs() * active)
            dormant.mul_(DORMANT_BETA)
            dormant.add_((1.0 - DORMANT_BETA) * g.abs() * inactive)

    @torch.no_grad()
    def _zero_optimizer_positions_(self, optimizer: torch.optim.Optimizer, weight: nn.Parameter, changed: torch.Tensor) -> None:
        state = optimizer.state.get(weight, {})
        for value in state.values():
            if torch.is_tensor(value) and value.shape == weight.shape:
                value[changed] = 0

    @torch.no_grad()
    def _promote_consolidated_(self, n: str, mask: torch.Tensor) -> Dict[str, int]:
        consolidated = getattr(self, f"consolidated_{n}")
        streak = getattr(self, f"streak_{n}")
        utility = getattr(self, f"utility_{n}")
        active = mask > 0.5
        if not self.consolidation_enabled:
            consolidated.zero_(); streak.zero_()
            return {"active": int(active.sum()), "consolidated": 0, "newly_consolidated": 0}
        vals = utility[active]
        if vals.numel() == 0:
            return {"active": 0, "consolidated": int(consolidated.sum()), "newly_consolidated": 0}
        k = max(1, int(round(vals.numel() * CONSOLIDATE_TOP_FRACTION)))
        threshold = torch.topk(vals, k=k, largest=True).values.min()
        strong = active & (utility >= threshold)
        streak[strong] += 1
        streak[active & ~strong] = 0
        before = consolidated.clone()
        consolidated |= active & (streak >= CONSOLIDATE_STREAK)
        newly = int((consolidated & ~before).sum())
        return {"active": int(active.sum()), "consolidated": int(consolidated.sum()), "newly_consolidated": newly}

    @torch.no_grad()
    def rewire_(self, optimizer: torch.optim.Optimizer, *, seed: int, epoch: int) -> Dict[str, object]:
        event: Dict[str, object] = {"epoch": epoch, "matrices": {}}
        for matrix_id, (n, w, mask) in enumerate(self._triples()):
            promotion = self._promote_consolidated_(n, mask)
            utility = getattr(self, f"utility_{n}")
            dormant = getattr(self, f"dormant_{n}")
            streak = getattr(self, f"streak_{n}")
            consolidated = getattr(self, f"consolidated_{n}")
            active = mask > 0.5
            active_count = int(active.sum())
            target_swap = max(1, int(round(active_count * REWIRE_FRACTION)))
            eligible = active & (~consolidated if self.consolidation_enabled else torch.ones_like(active, dtype=torch.bool))
            eligible_idx = torch.nonzero(eligible.view(-1), as_tuple=False).flatten()
            swap = min(target_swap, int(eligible_idx.numel()))
            if swap <= 0:
                event["matrices"][n] = {**promotion, "swapped": 0}
                continue

            flat_u = utility.view(-1)
            local = torch.topk(flat_u[eligible_idx], k=swap, largest=False, sorted=False).indices
            prune_idx = eligible_idx[local]
            flat_mask = mask.view(-1)
            flat_w = w.view(-1)
            flat_d = dormant.view(-1)
            flat_s = streak.view(-1)
            flat_c = consolidated.view(-1)
            flat_mask[prune_idx] = 0.0
            flat_w[prune_idx] = 0.0
            flat_d[prune_idx] = 0.0
            flat_s[prune_idx] = 0
            flat_c[prune_idx] = False

            inactive_idx = torch.nonzero(flat_mask < 0.5, as_tuple=False).flatten()
            # Newly pruned positions have dormant score reset to zero, preventing
            # immediate prune-and-regrow cancellation unless all candidates are zero.
            regrow_local = torch.topk(flat_d[inactive_idx], k=swap, largest=True, sorted=False).indices
            regrow_idx = inactive_idx[regrow_local]
            flat_mask[regrow_idx] = 1.0

            gen = torch.Generator(device=w.device)
            gen.manual_seed(seed * 100000 + epoch * 100 + matrix_id)
            active_std = float(w[mask > 0.5].std().item()) if bool((mask > 0.5).any()) else 0.01
            init_std = max(active_std * 0.01, 1e-5)
            flat_w[regrow_idx] = torch.randn(regrow_idx.numel(), generator=gen, device=w.device, dtype=w.dtype) * init_std
            flat_s[regrow_idx] = 0
            flat_c[regrow_idx] = False
            utility.view(-1)[regrow_idx] = 0.0
            dormant.view(-1)[regrow_idx] = 0.0

            changed = torch.zeros_like(mask, dtype=torch.bool).view(-1)
            changed[prune_idx] = True; changed[regrow_idx] = True
            changed = changed.view_as(mask)
            self._zero_optimizer_positions_(optimizer, w, changed)

            if int((mask > 0.5).sum()) != active_count:
                raise AssertionError(f"active count changed matrix={n}")
            event["matrices"][n] = {
                **promotion,
                "swapped": swap,
                "active_after": int((mask > 0.5).sum()),
                "consolidated_after": int(consolidated.sum()),
            }
        self.apply_masks_()
        self.rewire_events.append(event)
        return event

    @torch.no_grad()
    def plasticity_report(self) -> Dict[str, object]:
        out: Dict[str, object] = {"consolidation_enabled": self.consolidation_enabled, "events": len(self.rewire_events), "matrices": {}}
        total_active = total_consolidated = 0
        for n, _, mask in self._triples():
            active = int((mask > 0.5).sum())
            con = int(getattr(self, f"consolidated_{n}").sum())
            total_active += active; total_consolidated += con
            out["matrices"][n] = {"active": active, "consolidated": con}
        out["active_total"] = total_active
        out["consolidated_total"] = total_consolidated
        out["consolidated_fraction_of_active"] = total_consolidated / max(total_active, 1)
        out["rewire_events"] = self.rewire_events
        return out


def train_plastic_variant(model: PlasticQATP30RSNN, train: dm.Bank, monitor: dm.Bank, *, epochs: int, batch_size: int, seed: int, label: str) -> Tuple[PlasticQATP30RSNN, Dict[str, object]]:
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arch.LR, weight_decay=arch.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=arch.ETA_MIN)
    criterion = nn.SmoothL1Loss(beta=1.0)
    started = time.perf_counter()
    history: List[Dict[str, object]] = []
    checkpoints = {1, 50, 100, 150, 151, 160, 180, 200, 220, 240, 250, epochs}

    for epoch in range(1, epochs + 1):
        target = sweep.pruning_target(epoch, FINAL_SPARSITY)
        if target is not None:
            model.prune_synapses(target, optimizer)
        model.quant_enabled = epoch >= qmod.QAT_START_EPOCH
        model.proxy_enabled = epoch >= PLASTICITY_START
        model.train()
        running = 0.0; seen = 0
        for ids in arch.deterministic_batches(len(train.x), batch_size, seed, epoch):
            bx = train.x[ids].to(DEVICE); by = train.y_scaled[ids].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = criterion(pred, by)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss seed={seed} label={label} epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if model.proxy_enabled:
                model.update_plasticity_scores_()
            model.zero_pruned_gradients_()
            optimizer.step(); model.apply_masks_()
            running += float(loss.detach()) * len(ids); seen += len(ids)
        scheduler.step()

        if epoch in REWIRE_EPOCHS:
            event = model.rewire_(optimizer, seed=seed, epoch=epoch)
            print(f"NEUROPLASTIC_REWIRE seed={seed} label={label} {json.dumps(event, sort_keys=True)}", flush=True)
        if epoch in checkpoints:
            history.append({
                "epoch": epoch,
                "loss": running / max(seen, 1),
                "official_interpolate": dm.evaluate_raw(model, monitor),
                "sparsity": model.sparsity_report(),
                "plasticity": model.plasticity_report(),
            })
            print(f"CHECKPOINT seed={seed} label={label} epoch={epoch} loss={running/max(seen,1):.8f}", flush=True)

    model.quant_enabled = True; model.proxy_enabled = False
    elapsed = time.perf_counter() - started
    report = model.sparsity_report()
    for key in ("W_in", "W_rec", "W_out", "overall"):
        if abs(float(report[key]) - FINAL_SPARSITY) > 0.002:
            raise AssertionError(f"final sparsity mismatch {key}={report[key]}")
    if int(round(arch.count_parameters(model) * (1.0 - FINAL_SPARSITY))) != ACTIVE_EXPECTED:
        raise AssertionError("active parameter contract changed")
    return model, {
        "seed": seed,
        "label": label,
        "train_seconds": elapsed,
        "params": arch.count_parameters(model),
        "active_weights": ACTIVE_EXPECTED,
        "sparsity": report,
        "plasticity": model.plasticity_report(),
        "history": history,
    }


def attach_eval(result: Dict[str, object], model: nn.Module, banks: Dict[str, dm.Bank]) -> None:
    result["evaluations"] = {name: dm.evaluate_raw(model, bank) for name, bank in banks.items()}


def aggregate(runs: List[Dict[str, object]], banks: Sequence[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    seeds = sorted({int(r["seed"]) for r in runs})
    for label in LABELS:
        rows = [r for r in runs if r["label"] == label]
        if len(rows) != len(seeds):
            raise AssertionError(f"missing runs label={label}")
        entry: Dict[str, object] = {"train_seconds": {"mean": mean(float(r["train_seconds"]) for r in rows), "std": pstdev(float(r["train_seconds"]) for r in rows)}, "banks": {}}
        for bank in banks:
            entry["banks"][bank] = {}
            for metric in METRICS:
                vals = [float(r["evaluations"][bank][metric]) for r in rows]
                entry["banks"][bank][metric] = {"mean": mean(vals), "std": pstdev(vals), "min": min(vals), "max": max(vals)}
        if label != BASELINE:
            entry["consolidated_fraction_mean"] = mean(float(r["plasticity"]["consolidated_fraction_of_active"]) for r in rows)
        out[label] = entry
    return out


def paired(runs: List[Dict[str, object]], candidate: str, banks: Sequence[str]) -> Dict[str, object]:
    refs = {int(r["seed"]): r for r in runs if r["label"] == BASELINE}
    cands = {int(r["seed"]): r for r in runs if r["label"] == candidate}
    out: Dict[str, object] = {}
    for bank in banks:
        mae_delta=[]; strict1=[]; within5=[]; residual=[]; wins=0
        for seed in sorted(refs):
            a=refs[seed]["evaluations"][bank]; b=cands[seed]["evaluations"][bank]
            mae_delta.append(100.0*(float(b["mae"])/float(a["mae"])-1.0))
            strict1.append(100.0*(float(b["strict_1_0"])-float(a["strict_1_0"])))
            within5.append(100.0*(float(b["within_5_0"])-float(a["within_5_0"])))
            residual.append(100.0*(float(b["equation_residual_mean"])/float(a["equation_residual_mean"])-1.0))
            wins += int(float(b["mae"]) < float(a["mae"]))
        out[bank]={
            "mae_change_pct_mean":mean(mae_delta), "mae_change_pct_std":pstdev(mae_delta),
            "strict1_change_pp_mean":mean(strict1), "within5_change_pp_mean":mean(within5),
            "residual_change_pct_mean":mean(residual), "mae_wins":wins, "mae_losses":len(refs)-wins,
        }
    return out


def preflight() -> None:
    set_seed(1234)
    m = PlasticQATP30RSNN(consolidation=True)
    opt = torch.optim.AdamW(m.parameters(), lr=arch.LR)
    m.prune_synapses(FINAL_SPARSITY, opt)
    before = sum(int((mask > 0.5).sum()) for _, _, mask in m._triples())
    m.quant_enabled=True; m.proxy_enabled=True
    x=torch.randn(32, arch.IN_DIM); y=torch.randn(32, arch.OUT_DIM)
    loss=nn.SmoothL1Loss()(m(x),y); loss.backward(); m.update_plasticity_scores_()
    dormant_signal=sum(float(getattr(m,f"dormant_{n}").sum()) for n,_,_ in m._triples())
    if dormant_signal <= 0.0:
        raise AssertionError("no dormant gradient signal")
    # Force two consolidation checkpoints before rewiring to prove protection path.
    for n,_,mask in m._triples():
        u=getattr(m,f"utility_{n}"); u.copy_(torch.arange(u.numel(),dtype=u.dtype).reshape_as(u))
        m._promote_consolidated_(n,mask); m._promote_consolidated_(n,mask)
    event=m.rewire_(opt,seed=1234,epoch=160)
    after=sum(int((mask > 0.5).sum()) for _,_,mask in m._triples())
    if before != after or after != ACTIVE_EXPECTED:
        raise AssertionError(f"active count changed before={before} after={after}")
    exported=qmod.Int8WeightRSNN(m)
    if exported.storage_report()["logical_active_weights"] != ACTIVE_EXPECTED:
        raise AssertionError("INT8 export active count mismatch")
    print(f"NEUROPLASTICITY_PREFLIGHT_PASS active={after} dormant_signal={dormant_signal:.6g} event={json.dumps(event,sort_keys=True)}",flush=True)


def make_summary(agg: Dict[str, object], comparisons: Dict[str, object]) -> str:
    official="OFFICIAL_INTERPOLATE"
    lines=[
        "P30 INT8-QAT NEUROPLASTICITY ABLATION — DEEPMIND linear_2d","",
        "Computational analogue only; not a literal biological synapse model.",
        f"Rewire epochs={REWIRE_EPOCHS}; swap fraction={REWIRE_FRACTION:.3f}; consolidation top fraction={CONSOLIDATE_TOP_FRACTION:.3f}; streak={CONSOLIDATE_STREAK}.","",
    ]
    for label in LABELS:
        b=agg[label]["banks"][official]
        extra=""
        if label!=BASELINE:
            extra=f" consolidated={100.0*agg[label].get('consolidated_fraction_mean',0.0):.2f}%"
        lines.append(f"{label}: MAE={b['mae']['mean']:.8f} RMSE={b['rmse']['mean']:.8f} strict1={100*b['strict_1_0']['mean']:.4f}% within5={100*b['within_5_0']['mean']:.4f}% train_s={agg[label]['train_seconds']['mean']:.3f}{extra}")
    lines += ["", "Paired vs baseline:"]
    for cand, banks in comparisons.items():
        for bank,row in banks.items():
            lines.append(f"{cand} {bank}: MAE_delta={row['mae_change_pct_mean']:+.4f}% strict1_delta={row['strict1_change_pp_mean']:+.4f}pp within5_delta={row['within5_change_pp_mean']:+.4f}pp residual_delta={row['residual_change_pct_mean']:+.4f}% wins={row['mae_wins']}/{row['mae_wins']+row['mae_losses']}")
    return "\n".join(lines)+"\n"


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--preflight",action="store_true")
    p.add_argument("--train-size",type=int,default=8192); p.add_argument("--eval-size",type=int,default=2048)
    p.add_argument("--ill-pool-size",type=int,default=4096); p.add_argument("--ill-size",type=int,default=1024)
    p.add_argument("--epochs",type=int,default=300); p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--seeds",default="11,22,33,44,55"); p.add_argument("--output-dir",default="deepmind-p30-int8-neuroplasticity-output")
    args=p.parse_args(); torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    if args.preflight: preflight(); return
    if args.epochs < 300: raise ValueError("full controlled test requires 300 epochs")
    seeds=[int(s) for s in args.seeds.split(",") if s.strip()];
    if len(seeds)<3: raise ValueError("at least 3 seeds required")
    train,banks,manifest=qmod.build_banks(args)
    print("NEUROPLASTICITY_FAIRNESS_CONTRACT same DeepMind banks/seeds/init/optimizer/pruning/QAT; only post-pruning structural plasticity differs",flush=True)
    print("DATA_CONTRACT pinned google-deepmind/mathematics_dataset linear_2d; project synthetic data=0",flush=True)
    runs:List[Dict[str,object]]=[]
    for seed in seeds:
        set_seed(seed); base=qmod.QATP30RSNN(); init=clone_state(base)
        set_seed(seed); rw=PlasticQATP30RSNN(consolidation=False); assert_same_initialization(init,clone_state(rw),f"seed={seed} rewire")
        set_seed(seed); co=PlasticQATP30RSNN(consolidation=True); assert_same_initialization(init,clone_state(co),f"seed={seed} consolidated")

        base,br=qmod.train_qat_variant(base,train,banks["OFFICIAL_INTERPOLATE"],epochs=args.epochs,batch_size=args.batch_size,seed=seed)
        br["label"]=BASELINE; be=qmod.Int8WeightRSNN(base).to(DEVICE); attach_eval(br,be,banks); runs.append(br)

        rw,rr=train_plastic_variant(rw,train,banks["OFFICIAL_INTERPOLATE"],epochs=args.epochs,batch_size=args.batch_size,seed=seed,label=REWIRE)
        rwe=qmod.Int8WeightRSNN(rw).to(DEVICE); attach_eval(rr,rwe,banks); runs.append(rr)

        co,cr=train_plastic_variant(co,train,banks["OFFICIAL_INTERPOLATE"],epochs=args.epochs,batch_size=args.batch_size,seed=seed,label=CONSOLIDATED)
        coe=qmod.Int8WeightRSNN(co).to(DEVICE); attach_eval(cr,coe,banks); runs.append(cr)
        print(f"SEED_COMPLETE seed={seed}",flush=True)

    bank_names=list(banks.keys()); agg=aggregate(runs,bank_names)
    comps={REWIRE:paired(runs,REWIRE,bank_names),CONSOLIDATED:paired(runs,CONSOLIDATED,bank_names)}
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    results={
        "experiment":"P30_INT8_QAT_NEUROPLASTICITY_ABLATION","device":str(DEVICE),"seeds":seeds,
        "epochs":args.epochs,"batch_size":args.batch_size,"final_sparsity":FINAL_SPARSITY,
        "rewire_epochs":REWIRE_EPOCHS,"rewire_fraction":REWIRE_FRACTION,
        "utility_beta":UTILITY_BETA,"dormant_beta":DORMANT_BETA,
        "consolidate_top_fraction":CONSOLIDATE_TOP_FRACTION,"consolidate_streak":CONSOLIDATE_STREAK,
        "runs":runs,"aggregate":agg,"paired_vs_baseline":comps,
    }
    (out/"results.json").write_text(json.dumps(results,indent=2,sort_keys=True),encoding="utf-8")
    manifest["neuroplasticity"]={"computational_analogue":True,"uses_test_labels":False,"active_weight_budget":ACTIVE_EXPECTED}
    (out/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding="utf-8")
    summary=make_summary(agg,comps); (out/"SUMMARY.txt").write_text(summary,encoding="utf-8")
    print("\n"+summary,flush=True); print("NEUROPLASTICITY_EXPERIMENT_COMPLETE",flush=True)

if __name__=="__main__": main()
