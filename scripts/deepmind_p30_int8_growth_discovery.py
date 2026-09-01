#!/usr/bin/env python3
"""Growth/discovery ablation for the adopted P30 INT8-QAT RSNN.

Hypothesis
----------
The fixed P30 budget may hide useful connections because every new connection in
prior rewiring experiments forced another one to be removed immediately. This
experiment lets the network grow after P30 pruning, tracks repeated appearance
of useful candidate synapses, then selects a final topology using a validation
bank only. The official/stress banks are untouched until the final evaluation.

Variants
--------
1) P30_INT8_QAT_BASELINE
   Existing adopted P30 INT8-QAT model, fixed T=25.
2) P30_INT8_QAT_GROWTH_DISCOVERY
   Starts from the same initialization and P30 pruning schedule. After pruning,
   dormant connections receive zero-forward gradient probes. At discovery
   checkpoints, the best dormant candidates are activated without deleting old
   connections. Each connection accumulates an appearance count plus utility.
   Growth stops if the candidate set stops changing. At epoch 260 a separate
   validation bank selects the smallest topology within 0.25% MAE of the best
   candidate budget, then the selected topology is fine-tuned to epoch 300.

Important
---------
- Same pinned DeepMind mathematics_dataset linear_2d source and 5 paired seeds.
- No official/stress/ill labels are used for growth or final topology selection.
- Same weight-only INT8 QAT recipe; activations/membrane/accumulators stay FP32.
- Growth can expand from P30 (70% density) up to at most 85% density in this test.
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
GROWTH = "P30_INT8_QAT_GROWTH_DISCOVERY"
LABELS = (BASELINE, GROWTH)
FINAL_SPARSITY = 0.30
P30_ACTIVE = 18816
DISCOVERY_EPOCHS = (160, 180, 200, 220, 240, 260)
SELECTION_EPOCH = 260
GROW_FRACTION_TOTAL = 0.025
EVIDENCE_TOP_FRACTION = 0.20
UTILITY_BETA = 0.95
DORMANT_BETA = 0.95
CONVERGENCE_NEW_FRACTION = 0.01
CONVERGENCE_STREAK = 3
VALIDATION_TOLERANCE_PCT = 0.25
CANDIDATE_DENSITIES = (0.70, 0.75, 0.80, 0.85)
METRICS = qmod.METRICS


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def clone_core(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k in ("W_in","W_rec","W_out","M_in","M_rec","M_out")}


def assert_same_core(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], context: str) -> None:
    for k in a:
        if not torch.equal(a[k], b[k]):
            raise AssertionError(f"initialization mismatch {context} {k}")


class GrowthDiscoveryRSNN(qmod.QATP30RSNN):
    def __init__(self) -> None:
        super().__init__()
        self.proxy_enabled = False
        self.discovery_initialized = False
        self.growth_stopped = False
        self.low_novelty_streak = 0
        self.discovery_events: List[Dict[str, object]] = []
        for n, w in (("in",self.W_in),("rec",self.W_rec),("out",self.W_out)):
            self.register_buffer(f"utility_{n}", torch.zeros_like(w))
            self.register_buffer(f"dormant_{n}", torch.zeros_like(w))
            self.register_buffer(f"appearance_{n}", torch.zeros_like(w, dtype=torch.int16))
            self.register_buffer(f"ever_active_{n}", torch.zeros_like(w, dtype=torch.bool))

    def _triples(self):
        return (("in",self.W_in,self.M_in),("rec",self.W_rec,self.M_rec),("out",self.W_out,self.M_out))

    def initialize_discovery_(self) -> None:
        if self.discovery_initialized:
            return
        for n, _, mask in self._triples():
            getattr(self, f"ever_active_{n}").copy_(mask > 0.5)
        self.discovery_initialized = True

    def _effective(self, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = weight * mask
        if self.proxy_enabled:
            # Zero contribution in forward, unit derivative for dormant positions.
            masked = masked + (weight - weight.detach()) * (1.0 - mask)
        if self.quant_enabled:
            masked = qmod.fake_quant_per_channel_ste(masked)
        return masked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_in=self._effective(self.W_in,self.M_in); w_rec=self._effective(self.W_rec,self.M_rec); w_out=self._effective(self.W_out,self.M_out)
        batch=x.shape[0]; mem=x.new_zeros(batch,self.hidden_dim); spikes=x.new_zeros(batch,self.hidden_dim); out_acc=x.new_zeros(batch,self.out_dim)
        syn=F.linear(x,w_in)
        for _ in range(self.time_steps):
            rec=F.linear(spikes,w_rec)
            mem=self.decay*mem+syn+rec
            spikes=arch.spike_fn(mem-self.threshold)
            mem=mem-spikes*self.threshold
            out_acc=out_acc+F.linear(spikes,w_out)
        return out_acc/float(self.time_steps)

    @torch.no_grad()
    def update_scores_(self) -> None:
        for n,w,mask in self._triples():
            if w.grad is None: continue
            g=w.grad.detach(); active=mask>0.5; inactive=~active
            u=getattr(self,f"utility_{n}"); d=getattr(self,f"dormant_{n}")
            u.mul_(UTILITY_BETA).add_((1-UTILITY_BETA)*(w.detach()*g).abs()*active)
            d.mul_(DORMANT_BETA).add_((1-DORMANT_BETA)*g.abs()*inactive)

    @staticmethod
    def _norm(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        vals=x[mask]
        if vals.numel()==0: return torch.zeros_like(x)
        denom=float(vals.mean().item())+1e-12
        return x/denom

    @torch.no_grad()
    def discover_and_grow_(self, optimizer: torch.optim.Optimizer, *, seed: int, epoch: int) -> Dict[str, object]:
        self.initialize_discovery_()
        event: Dict[str, object] = {"epoch":epoch,"growth_stopped_before":self.growth_stopped,"matrices":{}}
        total_evidence=0; total_new_evidence=0
        for matrix_id,(n,w,mask) in enumerate(self._triples()):
            active=mask>0.5; inactive=~active
            u=getattr(self,f"utility_{n}"); d=getattr(self,f"dormant_{n}"); app=getattr(self,f"appearance_{n}"); ever=getattr(self,f"ever_active_{n}")
            active_signal=self._norm(u,active); dormant_signal=self._norm(d,inactive)
            signal=torch.where(active,active_signal,dormant_signal)
            k_ev=max(1,int(round(signal.numel()*EVIDENCE_TOP_FRACTION)))
            top=torch.topk(signal.view(-1),k=k_ev,largest=True,sorted=False).indices
            flat_app=app.view(-1)
            new_count=int((flat_app[top]==0).sum().item())
            flat_app[top]+=1
            total_evidence+=k_ev; total_new_evidence+=new_count

            before=int(active.sum().item()); grown=0
            if not self.growth_stopped and bool(inactive.any()):
                grow=max(1,int(round(mask.numel()*GROW_FRACTION_TOTAL)))
                inactive_idx=torch.nonzero(inactive.view(-1),as_tuple=False).flatten()
                grow=min(grow,int(inactive_idx.numel()))
                flat_d=d.view(-1); local=torch.topk(flat_d[inactive_idx],k=grow,largest=True,sorted=False).indices
                add_idx=inactive_idx[local]
                flat_mask=mask.view(-1); flat_w=w.view(-1); flat_ever=ever.view(-1)
                flat_mask[add_idx]=1.0; flat_ever[add_idx]=True
                gen=torch.Generator(device=w.device); gen.manual_seed(seed*100000+epoch*100+matrix_id)
                active_std=float(w[active].std().item()) if bool(active.any()) else 0.01
                init_std=max(active_std*0.01,1e-5)
                flat_w[add_idx]=torch.randn(add_idx.numel(),generator=gen,device=w.device,dtype=w.dtype)*init_std
                # New connections start with clean optimizer state.
                state=optimizer.state.get(w,{})
                changed=torch.zeros_like(mask,dtype=torch.bool).view(-1); changed[add_idx]=True; changed=changed.view_as(mask)
                for value in state.values():
                    if torch.is_tensor(value) and value.shape==w.shape: value[changed]=0
                grown=int(add_idx.numel())
            event["matrices"][n]={"active_before":before,"grown":grown,"active_after":int((mask>0.5).sum().item()),"new_evidence":new_count,"evidence_slots":k_ev}

        novelty=total_new_evidence/max(total_evidence,1)
        if novelty<CONVERGENCE_NEW_FRACTION: self.low_novelty_streak+=1
        else: self.low_novelty_streak=0
        if self.low_novelty_streak>=CONVERGENCE_STREAK: self.growth_stopped=True
        event["novelty_fraction"]=novelty; event["low_novelty_streak"]=self.low_novelty_streak; event["growth_stopped_after"]=self.growth_stopped
        self.apply_masks_(); self.discovery_events.append(event); return event

    @torch.no_grad()
    def evidence_score(self, n: str, mask: torch.Tensor) -> torch.Tensor:
        app=getattr(self,f"appearance_{n}").float(); u=getattr(self,f"utility_{n}")
        active=mask>0.5; u_norm=self._norm(u,active)
        return app + 0.25*u_norm + 0.05*active.float()

    @torch.no_grad()
    def candidate_masks(self, density: float) -> Dict[str, torch.Tensor]:
        out={}
        for n,_,mask in self._triples():
            ever=getattr(self,f"ever_active_{n}")
            ids=torch.nonzero(ever.view(-1),as_tuple=False).flatten()
            target=min(int(round(mask.numel()*density)),int(ids.numel()))
            target=max(1,target)
            score=self.evidence_score(n,mask).view(-1)
            chosen=ids[torch.topk(score[ids],k=target,largest=True,sorted=False).indices]
            m=torch.zeros_like(mask).view(-1); m[chosen]=1.0; out[n]=m.view_as(mask)
        return out

    @torch.no_grad()
    def apply_candidate_masks_(self, masks: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer | None=None) -> None:
        for n,w,mask in self._triples():
            new=masks[n].to(mask.dtype); changed=(new!=mask)
            mask.copy_(new); w.mul_(mask)
            if optimizer is not None:
                state=optimizer.state.get(w,{})
                for value in state.values():
                    if torch.is_tensor(value) and value.shape==w.shape: value[changed]=0
        self.apply_masks_()

    def report(self) -> Dict[str, object]:
        mats={}; hist_total={}; active_total=0
        for n,_,mask in self._triples():
            app=getattr(self,f"appearance_{n}").detach().cpu().view(-1)
            vals,counts=torch.unique(app,return_counts=True)
            hist={str(int(v)):int(c) for v,c in zip(vals,counts)}
            mats[n]={"active":int((mask>0.5).sum()),"ever_active":int(getattr(self,f"ever_active_{n}").sum()),"appearance_histogram":hist}
            active_total+=mats[n]["active"]
            for k,v in hist.items(): hist_total[k]=hist_total.get(k,0)+v
        return {"active_total":active_total,"density":active_total/26880.0,"growth_stopped":self.growth_stopped,"events":self.discovery_events,"appearance_histogram_total":hist_total,"matrices":mats}


def build_all_banks(args):
    train,banks,manifest=qmod.build_banks(args)
    algebra=dm.prepare_deepmind(); seen=set()
    validation=dm.build_bank(algebra,name="VALIDATION_INTERPOLATE",kind="interpolate",count=args.validation_size,seed=12301,global_seen=seen)
    manifest["banks"][validation.name]=dm.bank_manifest(validation)
    manifest["selection_contract"]="VALIDATION_INTERPOLATE only; official/stress/ill never used for topology selection"
    return train,validation,banks,manifest


@torch.no_grad()
def validation_metrics_for_masks(model: GrowthDiscoveryRSNN, masks: Dict[str,torch.Tensor], validation: dm.Bank) -> Dict[str,float]:
    saved={n:mask.detach().clone() for n,_,mask in model._triples()}
    model.apply_candidate_masks_(masks,None)
    exported=qmod.Int8WeightRSNN(model).to(DEVICE)
    metrics=dm.evaluate_raw(exported,validation)
    model.apply_candidate_masks_(saved,None)
    return metrics


def train_growth(model: GrowthDiscoveryRSNN, train: dm.Bank, validation: dm.Bank, monitor: dm.Bank, *, epochs:int,batch_size:int,seed:int):
    model.to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=arch.LR,weight_decay=arch.WEIGHT_DECAY)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=arch.ETA_MIN)
    crit=nn.SmoothL1Loss(beta=1.0); started=time.perf_counter(); history=[]; selection={}
    for epoch in range(1,epochs+1):
        target=sweep.pruning_target(epoch,FINAL_SPARSITY)
        if target is not None: model.prune_synapses(target,opt)
        model.quant_enabled=epoch>=qmod.QAT_START_EPOCH
        if epoch>=qmod.QAT_START_EPOCH:
            model.initialize_discovery_(); model.proxy_enabled=True
        model.train(); running=0.0; seen=0
        for ids in arch.deterministic_batches(len(train.x),batch_size,seed,epoch):
            bx=train.x[ids].to(DEVICE); by=train.y_scaled[ids].to(DEVICE)
            opt.zero_grad(set_to_none=True); pred=model(bx); loss=crit(pred,by)
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite growth loss seed={seed} epoch={epoch}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            if model.proxy_enabled: model.update_scores_()
            model.zero_pruned_gradients_(); opt.step(); model.apply_masks_(); running+=float(loss.detach())*len(ids); seen+=len(ids)
        sch.step()

        if epoch in DISCOVERY_EPOCHS:
            event=model.discover_and_grow_(opt,seed=seed,epoch=epoch)
            print(f"GROWTH_DISCOVERY seed={seed} {json.dumps(event,sort_keys=True)}",flush=True)

        if epoch==SELECTION_EPOCH:
            current_density=model.report()["density"]
            candidates=[]
            for d in CANDIDATE_DENSITIES:
                if d<=current_density+1e-9:
                    masks=model.candidate_masks(d); m=validation_metrics_for_masks(model,masks,validation)
                    candidates.append({"density":d,"mae":float(m["mae"]),"metrics":m,"masks":masks})
            if not candidates: raise AssertionError("no selection candidates")
            best_mae=min(c["mae"] for c in candidates); cutoff=best_mae*(1.0+VALIDATION_TOLERANCE_PCT/100.0)
            eligible=[c for c in candidates if c["mae"]<=cutoff]
            chosen=min(eligible,key=lambda c:c["density"])
            model.apply_candidate_masks_(chosen["masks"],opt); model.growth_stopped=True
            selection={"best_validation_mae":best_mae,"tolerance_pct":VALIDATION_TOLERANCE_PCT,"chosen_density":chosen["density"],"chosen_validation_mae":chosen["mae"],"candidates":[{"density":c["density"],"mae":c["mae"],"metrics":c["metrics"]} for c in candidates]}
            print(f"GROWTH_SELECTION seed={seed} {json.dumps(selection,sort_keys=True)}",flush=True)

        if epoch in {1,50,100,150,160,180,200,220,240,260,280,epochs}:
            history.append({"epoch":epoch,"loss":running/max(seen,1),"official_monitor":dm.evaluate_raw(model,monitor),"report":model.report()})

    model.quant_enabled=True; model.proxy_enabled=False
    return model,{"seed":seed,"label":GROWTH,"train_seconds":time.perf_counter()-started,"history":history,"selection":selection,"report":model.report()}


def attach_eval(result:Dict[str,object],model:nn.Module,banks:Dict[str,dm.Bank])->None:
    result["evaluations"]={name:dm.evaluate_raw(model,bank) for name,bank in banks.items()}


def aggregate(runs:List[Dict[str,object]],bank_names:Sequence[str])->Dict[str,object]:
    out={}
    for label in LABELS:
        rows=[r for r in runs if r["label"]==label]; entry={"banks":{},"train_seconds":{"mean":mean(float(r["train_seconds"]) for r in rows),"std":pstdev(float(r["train_seconds"]) for r in rows)}}
        for bank in bank_names:
            entry["banks"][bank]={}
            for metric in METRICS:
                vals=[float(r["evaluations"][bank][metric]) for r in rows]
                entry["banks"][bank][metric]={"mean":mean(vals),"std":pstdev(vals),"min":min(vals),"max":max(vals)}
        out[label]=entry
    return out


def paired(runs:List[Dict[str,object]],bank_names:Sequence[str])->Dict[str,object]:
    base={int(r["seed"]):r for r in runs if r["label"]==BASELINE}; grow={int(r["seed"]):r for r in runs if r["label"]==GROWTH}; out={}
    for bank in bank_names:
        mae=[]; strict=[]; within=[]; residual=[]; wins=0
        for seed in sorted(base):
            a=base[seed]["evaluations"][bank]; b=grow[seed]["evaluations"][bank]
            mae.append(100*(float(b["mae"])/float(a["mae"])-1)); strict.append(100*(float(b["strict_1_0"])-float(a["strict_1_0"]))); within.append(100*(float(b["within_5_0"])-float(a["within_5_0"]))); residual.append(100*(float(b["equation_residual_mean"])/float(a["equation_residual_mean"])-1)); wins+=int(float(b["mae"])<float(a["mae"]))
        out[bank]={"mae_change_pct_mean":mean(mae),"mae_change_pct_std":pstdev(mae),"strict1_change_pp_mean":mean(strict),"within5_change_pp_mean":mean(within),"residual_change_pct_mean":mean(residual),"mae_wins":wins,"mae_losses":len(base)-wins}
    return out


def preflight() -> None:
    set_seed(1234); m=GrowthDiscoveryRSNN(); opt=torch.optim.AdamW(m.parameters(),lr=arch.LR,weight_decay=arch.WEIGHT_DECAY); m.prune_synapses(0.30,opt); m.initialize_discovery_()
    before=m.report()["active_total"]; m.proxy_enabled=True; m.quant_enabled=True
    x=torch.randn(32,arch.IN_DIM); y=m(x).sum(); y.backward(); m.update_scores_(); event=m.discover_and_grow_(opt,seed=1234,epoch=160); after=m.report()["active_total"]
    if before!=P30_ACTIVE: raise AssertionError((before,P30_ACTIVE))
    if after<=before: raise AssertionError("growth did not increase active connections")
    if after>26880: raise AssertionError("growth exceeded dense topology")
    print(f"GROWTH_DISCOVERY_PREFLIGHT_PASS before={before} after={after} novelty={event['novelty_fraction']:.6f}",flush=True)


def make_summary(agg,comp,runs)->str:
    off="OFFICIAL_INTERPOLATE"; lines=["P30 INT8-QAT GROWTH DISCOVERY — DEEPMIND linear_2d","","Selection uses VALIDATION_INTERPOLATE only; official/stress/ill are final evaluation only."]
    for label in LABELS:
        b=agg[label]["banks"][off]; lines.append(f"{label}: MAE={b['mae']['mean']:.8f} RMSE={b['rmse']['mean']:.8f} strict1={100*b['strict_1_0']['mean']:.4f}% within5={100*b['within_5_0']['mean']:.4f}%")
    lines.append("")
    for bank,row in comp.items(): lines.append(f"{bank}: MAE_delta={row['mae_change_pct_mean']:+.4f}% strict1_delta={row['strict1_change_pp_mean']:+.4f}pp within5_delta={row['within5_change_pp_mean']:+.4f}pp residual_delta={row['residual_change_pct_mean']:+.4f}% wins={row['mae_wins']}/{row['mae_wins']+row['mae_losses']}")
    lines.append("")
    for r in runs:
        if r["label"]==GROWTH:
            lines.append(f"seed={r['seed']} chosen_density={r['selection'].get('chosen_density')} active_final={r['report']['active_total']} growth_stopped={r['report']['growth_stopped']} appearance_hist={json.dumps(r['report']['appearance_histogram_total'],sort_keys=True)}")
    return "\n".join(lines)+"\n"


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--preflight",action="store_true"); p.add_argument("--train-size",type=int,default=8192); p.add_argument("--validation-size",type=int,default=1024); p.add_argument("--eval-size",type=int,default=2048); p.add_argument("--ill-pool-size",type=int,default=4096); p.add_argument("--ill-size",type=int,default=1024); p.add_argument("--epochs",type=int,default=300); p.add_argument("--batch-size",type=int,default=256); p.add_argument("--seeds",default="11,22,33,44,55"); p.add_argument("--output-dir",default="deepmind-p30-int8-growth-discovery-output"); args=p.parse_args()
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    if args.preflight: preflight(); return
    if args.epochs<300: raise ValueError("full controlled test requires 300 epochs")
    seeds=[int(s) for s in args.seeds.split(",") if s.strip()];
    train,validation,banks,manifest=build_all_banks(args); runs=[]
    print("GROWTH_FAIRNESS_CONTRACT same init/data/batches/optimizer/P30/QAT; growth-discovery and validation-only topology selection are the only changes",flush=True)
    print("DATA_CONTRACT pinned google-deepmind/mathematics_dataset linear_2d; project synthetic data=0",flush=True)
    for seed in seeds:
        set_seed(seed); base=qmod.QATP30RSNN(); init=clone_core(base)
        set_seed(seed); grow=GrowthDiscoveryRSNN(); assert_same_core(init,clone_core(grow),f"seed={seed}")
        base,br=qmod.train_qat_variant(base,train,banks["OFFICIAL_INTERPOLATE"],epochs=args.epochs,batch_size=args.batch_size,seed=seed); br["label"]=BASELINE; be=qmod.Int8WeightRSNN(base).to(DEVICE); attach_eval(br,be,banks); runs.append(br)
        grow,gr=train_growth(grow,train,validation,banks["OFFICIAL_INTERPOLATE"],epochs=args.epochs,batch_size=args.batch_size,seed=seed); ge=qmod.Int8WeightRSNN(grow).to(DEVICE); attach_eval(gr,ge,banks); runs.append(gr); print(f"SEED_COMPLETE seed={seed}",flush=True)
    bank_names=list(banks.keys()); agg=aggregate(runs,bank_names); comp=paired(runs,bank_names)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); result={"experiment":"P30_INT8_QAT_GROWTH_DISCOVERY","device":str(DEVICE),"seeds":seeds,"epochs":args.epochs,"batch_size":args.batch_size,"discovery_epochs":DISCOVERY_EPOCHS,"grow_fraction_total":GROW_FRACTION_TOTAL,"evidence_top_fraction":EVIDENCE_TOP_FRACTION,"convergence_new_fraction":CONVERGENCE_NEW_FRACTION,"convergence_streak":CONVERGENCE_STREAK,"validation_tolerance_pct":VALIDATION_TOLERANCE_PCT,"candidate_densities":CANDIDATE_DENSITIES,"runs":runs,"aggregate":agg,"paired_vs_baseline":comp}
    (out/"results.json").write_text(json.dumps(result,indent=2,sort_keys=True),encoding="utf-8"); manifest["growth_discovery"]={"uses_official_for_selection":False,"uses_validation_only":True,"initial_active":P30_ACTIVE}; (out/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding="utf-8"); summary=make_summary(agg,comp,runs); (out/"SUMMARY.txt").write_text(summary,encoding="utf-8"); print("\n"+summary,flush=True); print("GROWTH_DISCOVERY_EXPERIMENT_COMPLETE",flush=True)

if __name__=="__main__": main()
