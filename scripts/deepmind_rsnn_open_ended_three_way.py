#!/usr/bin/env python3
"""Open-ended three-way RSNN experiment.

Variants run as separate GitHub matrix jobs but use the same deterministic
DeepMind data stream seeds and the same fixed held-out banks:
  adopted_p30 : adopted P30 + INT8-QAT + fixed T=25
  open_growth : starts from the same P30 recipe, then unconstrained synaptic
                growth/selection with persistence + contribution scoring
  dense_long  : no pruning, long-run Dense RSNN + INT8-QAT + fixed T=25

The open-ended phase has no fixed epoch/data limit. Stopping is governed by
validation stability/deterioration, topology stability for open_growth, or the
hard wall-clock training budget. Official/stress/ill banks are never used to
choose topology or stopping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

import pruned_snn_vs_mlp as arch
import deepmind_pruned_snn_stress as dm
import deepmind_rsnn_pruning_sweep as sweep
import deepmind_p30_int8_ablation as qmod

DEVICE = torch.device("cpu")
SEED = 11
FINAL_SPARSITY = 0.30
P30_ACTIVE = 18816
TOTAL_WEIGHTS = 26880
WARMUP_EPOCHS = 150
STREAM_CHUNK = 2048
PASSES_PER_CHUNK = 4
TRAIN_BUDGET_DEFAULT = 13200.0  # 3h40m; leaves room for final evaluation/upload under 4h.
VAL_REL_IMPROVEMENT = 0.001      # 0.1% meaningful improvement.
PLATEAU_PATIENCE = 12
DEGRADE_MARGIN = 0.01            # 1% above best.
DEGRADE_PATIENCE = 4
MIN_OPEN_CYCLES = 12
GROWTH_NOVELTY = 0.01
GROWTH_STABLE_CYCLES = 3
IMPORTANT_FRACTION = 0.20
PROTECT_FRACTION = 0.02
PRUNE_FRACTION = 0.02
SELECTION_NOVELTY = 0.01
SELECTION_STABLE_CYCLES = 3
UTILITY_BETA = 0.95
REGROW_INIT_SCALE = 0.01


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def bank_fingerprint(bank: dm.Bank) -> str:
    h = hashlib.sha256()
    h.update(bank.x.detach().cpu().numpy().tobytes())
    h.update(bank.y_raw.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def bank_keys(bank: dm.Bank) -> set:
    out = set()
    x = bank.x.detach().cpu().numpy(); y = bank.y_raw.detach().cpu().numpy()
    for a, b in zip(x, y):
        out.add(tuple(np.round(a.astype(np.float64), 10)) + tuple(np.round(b.astype(np.float64), 8)))
    return out


def count_active(model: arch.PrunedBrainSpikeNet) -> int:
    return sum(int((m > 0.5).sum().item()) for m in (model.M_in, model.M_rec, model.M_out))


def matrix_triplets(model):
    return (("in", model.W_in, model.M_in), ("rec", model.W_rec, model.M_rec), ("out", model.W_out, model.M_out))


def zero_optimizer_positions_(optimizer, param, changed: torch.Tensor) -> None:
    state = optimizer.state.get(param, {})
    for value in state.values():
        if torch.is_tensor(value) and value.shape == param.shape:
            value[changed] = 0


def train_batches(model, bank: dm.Bank, optimizer, criterion, *, seed: int, cycle: int, passes: int, utility_cb=None) -> float:
    model.train(); total = 0.0; seen = 0
    for p in range(passes):
        for ids in arch.deterministic_batches(len(bank.x), 256, seed + 100003 * cycle + p, p + 1):
            bx = bank.x[ids].to(DEVICE); by = bank.y_scaled[ids].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx); loss = criterion(pred, by)
            if not torch.isfinite(loss): raise RuntimeError("non-finite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if utility_cb is not None: utility_cb()
            model.zero_pruned_gradients_(); optimizer.step(); model.apply_masks_()
            total += float(loss.detach()) * len(ids); seen += len(ids)
    return total / max(seen, 1)


@torch.no_grad()
def eval_mae(model, bank: dm.Bank) -> float:
    return float(dm.evaluate_raw(model, bank)["mae"])


def bank_loss(model, bank: dm.Bank, criterion, *, require_grad=False) -> torch.Tensor:
    if not require_grad: model.eval()
    vals = []
    for start in range(0, len(bank.x), 512):
        bx = bank.x[start:start+512].to(DEVICE); by = bank.y_scaled[start:start+512].to(DEVICE)
        vals.append(criterion(model(bx), by) * len(bx))
    return torch.stack(vals).sum() / len(bank.x)


class OpenGrowthRSNN(qmod.QATP30RSNN):
    """P30 start, no density ceiling, then persistence+contribution selection."""
    def __init__(self):
        super().__init__()
        self.phase = "growth"
        self.structural_cycle = 0
        self.growth_streak = 0
        self.selection_streak = 0
        self.selection_cycles = 0
        self.topology_stable = False
        self.prev_important: set[int] | None = None
        self.events: List[Dict[str, object]] = []
        for n, w, _ in matrix_triplets(self):
            self.register_buffer(f"utility_{n}", torch.zeros_like(w))
            self.register_buffer(f"appearance_{n}", torch.zeros_like(w, dtype=torch.int32))
            self.register_buffer(f"protected_{n}", torch.zeros_like(w, dtype=torch.bool))
            self.register_buffer(f"cooldown_{n}", torch.zeros_like(w, dtype=torch.int8))

    @torch.no_grad()
    def update_train_utility_(self) -> None:
        for n, w, mask in matrix_triplets(self):
            if w.grad is None: continue
            active = mask > 0.5
            s = (w.detach() * w.grad.detach()).abs()
            u = getattr(self, f"utility_{n}")
            u.mul_(UTILITY_BETA).add_((1.0-UTILITY_BETA) * s * active).mul_(active)

    def shadow_scores(self, bank: dm.Bank, criterion) -> Dict[str, torch.Tensor]:
        """Gradient scores for active and dormant zero-valued positions without changing predictions."""
        saved = {n: m.detach().clone() for n, _, m in matrix_triplets(self)}
        self.zero_grad(set_to_none=True)
        with torch.no_grad():
            for _, _, m in matrix_triplets(self): m.fill_(1.0)
        loss = bank_loss(self, bank, criterion, require_grad=True)
        loss.backward()
        scores = {}
        for n, w, _ in matrix_triplets(self):
            if w.grad is None: raise AssertionError("missing shadow gradient")
            scores[n] = w.grad.detach().abs().clone()
        with torch.no_grad():
            for n, _, m in matrix_triplets(self): m.copy_(saved[n])
        self.zero_grad(set_to_none=True); self.apply_masks_()
        return scores

    @torch.no_grad()
    def grow_(self, optimizer, shadow: Dict[str, torch.Tensor], *, seed: int) -> Dict[str, int]:
        total_new = 0; by_matrix = {}
        for mid, (n, w, mask) in enumerate(matrix_triplets(self)):
            cd = getattr(self, f"cooldown_{n}"); cd[cd > 0] -= 1
            active = mask > 0.5; dormant = (~active) & (cd <= 0)
            if not bool(dormant.any()): by_matrix[n] = 0; continue
            active_scores = shadow[n][active]
            threshold = float(torch.quantile(active_scores, 0.25).item()) * 0.5 if active_scores.numel() else 0.0
            chosen = dormant & (shadow[n] >= threshold)
            ids = torch.nonzero(chosen.view(-1), as_tuple=False).flatten()
            if ids.numel() == 0: by_matrix[n] = 0; continue
            std = float(w.detach()[active].std().item()) if bool(active.any()) else 0.01
            step = max(std * REGROW_INIT_SCALE, 1e-5)
            grad_flat = shadow[n].view(-1)[ids]
            gen = torch.Generator(device=w.device); gen.manual_seed(seed + 1009*mid + self.structural_cycle*100003)
            jitter = 0.5 + torch.rand(ids.numel(), generator=gen, device=w.device, dtype=w.dtype)
            # Magnitude from active scale; sign is randomized because shadow stores abs-grad only.
            signs = torch.where(torch.rand(ids.numel(), generator=gen, device=w.device) > 0.5, 1.0, -1.0).to(w.dtype)
            w.view(-1)[ids] = signs * jitter * step
            mask.view(-1)[ids] = 1.0
            changed = torch.zeros_like(mask, dtype=torch.bool).view(-1); changed[ids] = True
            zero_optimizer_positions_(optimizer, w, changed.view_as(mask))
            total_new += int(ids.numel()); by_matrix[n] = int(ids.numel())
        self.apply_masks_(); by_matrix["total"] = total_new
        return by_matrix

    def contribution_scores(self, bank: dm.Bank, criterion) -> Dict[str, torch.Tensor]:
        self.zero_grad(set_to_none=True)
        loss = bank_loss(self, bank, criterion, require_grad=True); loss.backward()
        out = {}
        for n, w, mask in matrix_triplets(self):
            out[n] = (w.detach() * w.grad.detach()).abs() * (mask > 0.5)
        self.zero_grad(set_to_none=True)
        return out

    def _global_arrays(self, contrib: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[str,int,int]]]:
        scores=[]; active=[]; protected=[]; spans=[]; start=0
        for n, _, mask in matrix_triplets(self):
            app = getattr(self, f"appearance_{n}").to(torch.float32)
            c = contrib[n]
            act = mask > 0.5
            if bool(act.any()):
                av = c[act]; med = torch.median(av).clamp_min(1e-12)
                cn = c / (c + med)
            else: cn = torch.zeros_like(c)
            p = app / max(self.selection_cycles + 1, 1)
            combined = torch.sqrt((p + 1e-6) * (cn + 1e-6))
            flat = combined.view(-1); scores.append(flat); active.append(act.view(-1)); protected.append(getattr(self, f"protected_{n}").view(-1))
            end=start+flat.numel(); spans.append((n,start,end)); start=end
        return torch.cat(scores), torch.cat(active), torch.cat(protected), spans

    @torch.no_grad()
    def _apply_global_mask(self, ids: torch.Tensor, spans, attr: str, value) -> None:
        for n, start, end in spans:
            local = ids[(ids >= start) & (ids < end)] - start
            if local.numel(): getattr(self, attr.format(n=n)).view(-1)[local] = value

    def structural_step(self, optimizer, structure_bank: dm.Bank, criterion, *, seed: int) -> Dict[str, object]:
        self.structural_cycle += 1
        event: Dict[str, object] = {"cycle":self.structural_cycle,"phase":self.phase,"active_before":count_active(self)}
        shadow = self.shadow_scores(structure_bank, criterion)

        if self.phase == "growth":
            grown = self.grow_(optimizer, shadow, seed=seed)
            novelty = grown["total"] / TOTAL_WEIGHTS
            self.growth_streak = self.growth_streak + 1 if novelty < GROWTH_NOVELTY else 0
            event.update({"grown":grown,"growth_novelty":novelty,"growth_streak":self.growth_streak})
            if self.growth_streak >= GROWTH_STABLE_CYCLES:
                self.phase = "selection"
                event["transition"] = "growth_to_selection"
            event["active_after"] = count_active(self); self.events.append(event); return event

        if self.phase != "selection":
            event["active_after"] = count_active(self); self.events.append(event); return event

        self.selection_cycles += 1
        # Allow any useful dormant candidates to refill; there is no density ceiling.
        grown = self.grow_(optimizer, shadow, seed=seed)
        contrib = self.contribution_scores(structure_bank, criterion)

        # Appearance = repeated membership among the most useful 20% active synapses.
        for n, _, mask in matrix_triplets(self):
            active = mask > 0.5; vals = contrib[n][active]
            if vals.numel():
                k=max(1,int(round(vals.numel()*IMPORTANT_FRACTION)))
                thr=torch.topk(vals,k=k,largest=True).values.min()
                getattr(self,f"appearance_{n}")[active & (contrib[n] >= thr)] += 1

        combined, active_g, protected_g, spans = self._global_arrays(contrib)
        active_ids = torch.nonzero(active_g, as_tuple=False).flatten()
        kimp=max(1,int(round(active_ids.numel()*IMPORTANT_FRACTION)))
        important_ids = active_ids[torch.topk(combined[active_ids],k=kimp,largest=True).indices]
        important_set = set(int(x) for x in important_ids.cpu().tolist())
        if self.prev_important is None:
            novelty=1.0
        else:
            novelty=len(important_set-self.prev_important)/max(len(important_set),1)
        self.prev_important=important_set
        self.selection_streak = self.selection_streak + 1 if novelty < SELECTION_NOVELTY else 0
        event.update({"grown":grown,"important_novelty":novelty,"selection_streak":self.selection_streak})

        if self.selection_streak >= SELECTION_STABLE_CYCLES:
            self.phase="final"; self.topology_stable=True
            event["transition"]="selection_to_final"
            event["active_after"]=count_active(self); self.events.append(event); return event

        # Protect best 2% and remove worst 2% among non-protected active synapses.
        eligible = active_g & (~protected_g)
        eligible_ids=torch.nonzero(eligible,as_tuple=False).flatten()
        nprotect=min(max(1,int(round(active_ids.numel()*PROTECT_FRACTION))), int(eligible_ids.numel()))
        top_ids=eligible_ids[torch.topk(combined[eligible_ids],k=nprotect,largest=True).indices]
        self._apply_global_mask(top_ids,spans,"protected_{n}",True)

        # Recompute eligible after protection.
        _, active_g, protected_g, spans = self._global_arrays(contrib)
        removable=active_g & (~protected_g); rem_ids=torch.nonzero(removable,as_tuple=False).flatten()
        nprune=min(max(1,int(round(active_ids.numel()*PRUNE_FRACTION))), int(rem_ids.numel()))
        bottom_ids=rem_ids[torch.topk(combined[rem_ids],k=nprune,largest=False).indices]

        base_loss=float(bank_loss(self,structure_bank,criterion,require_grad=False).detach())
        # Group ablation verification: how much validation loss changes if selected groups are removed.
        saved_masks={n:m.detach().clone() for n,_,m in matrix_triplets(self)}
        with torch.no_grad(): self._apply_global_mask(top_ids,spans,"M_{n}",0.0)
        top_loss=float(bank_loss(self,structure_bank,criterion,require_grad=False).detach())
        with torch.no_grad():
            for n,_,m in matrix_triplets(self): m.copy_(saved_masks[n])
            self._apply_global_mask(bottom_ids,spans,"M_{n}",0.0)
        bottom_loss=float(bank_loss(self,structure_bank,criterion,require_grad=False).detach())
        with torch.no_grad():
            for n,_,m in matrix_triplets(self): m.copy_(saved_masks[n])

        # Permanently prune bottom group; keep appearance history if it later regrows.
        for n,start,end in spans:
            local=bottom_ids[(bottom_ids>=start)&(bottom_ids<end)]-start
            if local.numel()==0: continue
            w=dict((a,b) for a,b,_ in matrix_triplets(self))[n]; mask=dict((a,c) for a,_,c in matrix_triplets(self))[n]
            changed=torch.zeros_like(mask,dtype=torch.bool).view(-1); changed[local]=True; changed=changed.view_as(mask)
            mask.view(-1)[local]=0.0; w.view(-1)[local]=0.0; getattr(self,f"cooldown_{n}").view(-1)[local]=1
            getattr(self,f"utility_{n}").view(-1)[local]=0.0
            zero_optimizer_positions_(optimizer,w,changed)
        self.apply_masks_()
        event.update({"protected_added":int(top_ids.numel()),"pruned":int(bottom_ids.numel()),
                      "ablation_top_loss_delta":top_loss-base_loss,"ablation_bottom_loss_delta":bottom_loss-base_loss,
                      "protected_total":sum(int(getattr(self,f"protected_{n}").sum()) for n,_,_ in matrix_triplets(self)),
                      "active_after":count_active(self)})
        self.events.append(event); return event


def build_fixed_banks(args):
    alg=dm.prepare_deepmind(); seen=set()
    warm=dm.build_bank(alg,name="WARMUP_TRAIN",kind="train",count=8192,seed=7301,global_seen=seen)
    struct=dm.build_bank(alg,name="STRUCTURE_VALIDATION",kind="train",count=2048,seed=7401,global_seen=seen)
    stop=dm.build_bank(alg,name="EARLYSTOP_VALIDATION",kind="train",count=2048,seed=7501,global_seen=seen)
    official=dm.build_bank(alg,name="OFFICIAL_INTERPOLATE",kind="interpolate",count=2048,seed=8301,global_seen=seen)
    stress10=dm.build_bank(alg,name="STRESS_ENTROPY_10",kind="entropy10",count=2048,seed=9301,global_seen=seen)
    stress12=dm.build_bank(alg,name="STRESS_ENTROPY_12",kind="entropy12",count=2048,seed=10301,global_seen=seen)
    ill_pool=dm.build_bank(alg,name="ILL_POOL_ENTROPY_12",kind="entropy12",count=4096,seed=11301,global_seen=seen)
    ids=torch.argsort(ill_pool.condition,descending=True)[:1024]; ill=dm.subset_bank(ill_pool,ids,"ILL_CONDITIONED")
    fixed={b.name:b for b in (warm,struct,stop,official,stress10,stress12,ill)}
    return alg,fixed,seen


def make_model(variant: str):
    set_seed(SEED)
    if variant=="open_growth": return OpenGrowthRSNN()
    return qmod.QATP30RSNN()


def warmup(model, variant: str, warm: dm.Bank, optimizer, criterion, start_time: float, budget: float):
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300,eta_min=arch.ETA_MIN)
    history=[]
    for epoch in range(1,WARMUP_EPOCHS+1):
        if time.monotonic()-start_time >= budget: return history,"time_budget_during_warmup"
        if variant!="dense_long":
            target=sweep.pruning_target(epoch,FINAL_SPARSITY)
            if target is not None: model.prune_synapses(target,optimizer)
        model.quant_enabled=False
        loss=train_batches(model,warm,optimizer,criterion,seed=SEED,cycle=epoch,passes=1)
        scheduler.step()
        if epoch in (1,50,100,150): history.append({"epoch":epoch,"loss":loss,"active":count_active(model)})
    if variant!="dense_long" and count_active(model)!=P30_ACTIVE: raise AssertionError(f"P30 warmup active={count_active(model)}")
    if variant=="dense_long" and count_active(model)!=TOTAL_WEIGHTS: raise AssertionError("dense warmup lost weights")
    model.quant_enabled=True
    return history,None


def run_variant(args):
    start=time.monotonic(); alg,fixed,fixed_seen=build_fixed_banks(args)
    fingerprints={k:bank_fingerprint(v) for k,v in fixed.items()}
    model=make_model(args.variant).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=arch.LR,weight_decay=arch.WEIGHT_DECAY)
    criterion=nn.SmoothL1Loss(beta=1.0)
    warm_hist,early_reason=warmup(model,args.variant,fixed["WARMUP_TRAIN"],optimizer,criterion,start,args.train_budget_seconds)
    if early_reason: raise RuntimeError(early_reason)

    # Same infinite deterministic prefix for every variant. Jobs may consume different prefix lengths if they converge earlier.
    stream_seen=set(fixed_seen)
    best_state=clone_state(model); best_mae=eval_mae(model,fixed["EARLYSTOP_VALIDATION"]); best_cycle=0
    best_after_stable_state=None; best_after_stable_mae=float("inf")
    plateau=0; degrade=0; cycles=0; examples_seen=8192*WARMUP_EPOCHS
    history=[]; stop_reason="time_budget"; final_phase_start=None
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=3,min_lr=1e-5)

    while True:
        elapsed=time.monotonic()-start
        if elapsed >= args.train_budget_seconds: stop_reason="time_budget"; break
        cycles += 1
        chunk=dm.build_bank(alg,name=f"STREAM_{cycles:05d}",kind="train",count=STREAM_CHUNK,
                            seed=15000+cycles,global_seen=stream_seen)
        utility_cb=model.update_train_utility_ if args.variant=="open_growth" and getattr(model,"phase","")!="final" else None
        loss=train_batches(model,chunk,optimizer,criterion,seed=SEED,cycle=1000+cycles,passes=PASSES_PER_CHUNK,utility_cb=utility_cb)
        examples_seen += STREAM_CHUNK*PASSES_PER_CHUNK

        event=None
        if args.variant=="open_growth" and not model.topology_stable:
            event=model.structural_step(optimizer,fixed["STRUCTURE_VALIDATION"],criterion,seed=SEED)
            if model.topology_stable and final_phase_start is None: final_phase_start=cycles

        val=eval_mae(model,fixed["EARLYSTOP_VALIDATION"]); scheduler.step(val)
        meaningful=val < best_mae*(1.0-VAL_REL_IMPROVEMENT)
        if val < best_mae:
            best_mae=val; best_state=clone_state(model); best_cycle=cycles
        plateau=0 if meaningful else plateau+1
        degrade=degrade+1 if val > best_mae*(1.0+DEGRADE_MARGIN) else 0

        stable_for_result = args.variant!="open_growth" or model.topology_stable
        if stable_for_result and val < best_after_stable_mae:
            best_after_stable_mae=val; best_after_stable_state=clone_state(model)

        row={"cycle":cycles,"train_loss":loss,"val_mae":val,"best_val_mae":best_mae,"lr":optimizer.param_groups[0]["lr"],
             "active":count_active(model),"elapsed_s":time.monotonic()-start,"event":event}
        history.append(row)
        print("OPEN_CYCLE "+json.dumps({k:v for k,v in row.items() if k!="event"},sort_keys=True),flush=True)
        if event is not None: print("STRUCTURE_EVENT "+json.dumps(event,sort_keys=True),flush=True)

        # For open_growth, topology stability must happen before validation stopping is allowed.
        can_stop = stable_for_result and cycles >= MIN_OPEN_CYCLES
        if args.variant=="open_growth" and final_phase_start is not None:
            can_stop = cycles-final_phase_start >= MIN_OPEN_CYCLES
        if can_stop and degrade >= DEGRADE_PATIENCE:
            stop_reason="persistent_validation_deterioration"; break
        if can_stop and plateau >= PLATEAU_PATIENCE:
            stop_reason="validation_plateau"; break

    # Prefer the best checkpoint after final topology stability; otherwise best overall if time expired first.
    chosen=best_after_stable_state if best_after_stable_state is not None else best_state
    model.load_state_dict(chosen,strict=True); model.quant_enabled=True; model.eval()
    exported=qmod.Int8WeightRSNN(model).to(DEVICE)
    final_banks=("OFFICIAL_INTERPOLATE","STRESS_ENTROPY_10","STRESS_ENTROPY_12","ILL_CONDITIONED")
    evaluations={name:dm.evaluate_raw(exported,fixed[name]) for name in final_banks}
    storage=exported.storage_report()
    protected_total=0
    if args.variant=="open_growth": protected_total=sum(int(getattr(model,f"protected_{n}").sum()) for n,_,_ in matrix_triplets(model))

    result={
        "experiment":"RSNN_OPEN_ENDED_THREE_WAY",
        "variant":args.variant,
        "seed":SEED,
        "deepmind_commit":dm.DM_COMMIT,
        "same_stream_contract":"all variants use identical fixed banks and deterministic STREAM seed=15000+cycle; each sees the same prefix, possibly different prefix length if it converges earlier",
        "train_budget_seconds":args.train_budget_seconds,
        "elapsed_training_and_generation_seconds":time.monotonic()-start,
        "stop_reason":stop_reason,
        "warmup_epochs":WARMUP_EPOCHS,
        "stream_cycles":cycles,
        "examples_seen_with_repeats":examples_seen,
        "best_validation_mae":best_mae,
        "best_cycle":best_cycle,
        "active_weights_final_chosen":count_active(model),
        "protected_total":protected_total,
        "topology_phase":getattr(model,"phase","fixed"),
        "topology_stable":bool(getattr(model,"topology_stable",True)),
        "bank_fingerprints":fingerprints,
        "storage":storage,
        "evaluations":evaluations,
        "warmup_history":warm_hist,
        "history":history,
        "structural_events":getattr(model,"events",[]),
    }
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(result,indent=2,sort_keys=True),encoding="utf-8")
    lines=[
        "RSNN OPEN-ENDED THREE-WAY",
        f"variant={args.variant}",f"stop_reason={stop_reason}",f"stream_cycles={cycles}",f"examples_seen={examples_seen}",
        f"best_validation_mae={best_mae:.8f}",f"active_weights={count_active(model)}",f"protected_total={protected_total}",
        f"topology_phase={getattr(model,'phase','fixed')}",f"topology_stable={getattr(model,'topology_stable',True)}",
    ]
    for name,m in evaluations.items(): lines.append(f"{name}: MAE={m['mae']:.8f} RMSE={m['rmse']:.8f} strict1={100*m['strict_1_0']:.4f}% within5={100*m['within_5_0']:.4f}%")
    (out/"SUMMARY.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (out/"bank_fingerprints.json").write_text(json.dumps(fingerprints,indent=2,sort_keys=True),encoding="utf-8")
    print("\n"+"\n".join(lines),flush=True)
    print("OPEN_ENDED_VARIANT_COMPLETE",flush=True)


def preflight(variant: str):
    set_seed(SEED); m=make_model(variant).to(DEVICE)
    opt=torch.optim.AdamW(m.parameters(),lr=arch.LR,weight_decay=arch.WEIGHT_DECAY)
    if variant!="dense_long":
        m.prune_synapses(FINAL_SPARSITY,opt)
        if count_active(m)!=P30_ACTIVE: raise AssertionError(count_active(m))
    else:
        if count_active(m)!=TOTAL_WEIGHTS: raise AssertionError(count_active(m))
    if variant=="open_growth":
        for n,_,_ in matrix_triplets(m):
            if not hasattr(m,f"appearance_{n}") or not hasattr(m,f"protected_{n}"): raise AssertionError("missing growth buffers")
    print(f"OPEN_ENDED_PREFLIGHT_PASS variant={variant} active={count_active(m)}",flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--variant",choices=["adopted_p30","open_growth","dense_long"],required=True)
    p.add_argument("--train-budget-seconds",type=float,default=TRAIN_BUDGET_DEFAULT)
    p.add_argument("--output-dir",default="rsnn-open-ended-output")
    p.add_argument("--preflight",action="store_true")
    args=p.parse_args(); torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    if args.preflight: preflight(args.variant); return
    run_variant(args)

if __name__=="__main__": main()
