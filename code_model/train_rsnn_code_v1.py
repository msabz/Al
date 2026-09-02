#!/usr/bin/env python3
"""Open-Growth RSNN Code V1 trainer.

Purpose: prove that the existing Open-Growth RSNN V2 ideas can be adapted to
prompt->Python generation while preserving quantization-aware training and sparse
structural growth. Training data must come from collect_real_python.py.

Training uses FP32 master weights/optimizer state plus fake INT8 quantization.
The exported deployment bundle stores INT8 weights and binary sparse masks.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

BYTE_VOCAB = 256
BOS = 256
EOS = 257
PAD = 258
VOCAB = 259


def fake_quant_int8(x: torch.Tensor, clip: float | None = None) -> torch.Tensor:
    if clip is None:
        mx = x.detach().abs().amax().clamp_min(1e-8)
    else:
        mx = torch.tensor(float(clip), device=x.device, dtype=x.dtype)
    scale = mx / 127.0
    q = torch.clamp(torch.round(x / scale), -127, 127)
    # Straight-through estimator: quantized forward, identity gradient.
    return x + (q * scale - x).detach()


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        # Fast-sigmoid surrogate derivative.
        surrogate = 1.0 / (1.0 + 2.0 * x.abs()).pow(2)
        return grad * surrogate


spike_fn = SpikeFn.apply


@dataclass
class SparseEntry:
    name: str
    param: nn.Parameter
    mask: torch.Tensor
    utility: torch.Tensor
    protected: torch.Tensor
    appearance: torch.Tensor


class SparseLIFLayer(nn.Module):
    def __init__(self, in_dim: int, hidden: int, initial_sparsity: float, mem_decay: float, syn_decay: float,
                 threshold: float, mem_clip: float) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.mem_decay = mem_decay
        self.syn_decay = syn_decay
        self.threshold = threshold
        self.mem_clip = mem_clip
        self.w_in = nn.Parameter(torch.empty(hidden, in_dim))
        self.w_rec = nn.Parameter(torch.empty(hidden, hidden))
        nn.init.kaiming_uniform_(self.w_in, a=math.sqrt(5))
        nn.init.orthogonal_(self.w_rec)
        self.w_rec.data.mul_(0.35)

        self.register_buffer("mask_in", torch.ones_like(self.w_in, dtype=torch.bool))
        self.register_buffer("mask_rec", torch.ones_like(self.w_rec, dtype=torch.bool))
        self.register_buffer("utility_in", torch.zeros_like(self.w_in))
        self.register_buffer("utility_rec", torch.zeros_like(self.w_rec))
        self.register_buffer("protected_in", torch.zeros_like(self.w_in, dtype=torch.bool))
        self.register_buffer("protected_rec", torch.zeros_like(self.w_rec, dtype=torch.bool))
        self.register_buffer("appearance_in", torch.zeros_like(self.w_in, dtype=torch.int32))
        self.register_buffer("appearance_rec", torch.zeros_like(self.w_rec, dtype=torch.int32))
        self._prune_initial(self.mask_in, self.w_in, initial_sparsity)
        self._prune_initial(self.mask_rec, self.w_rec, initial_sparsity)

    @staticmethod
    def _prune_initial(mask: torch.Tensor, param: nn.Parameter, sparsity: float) -> None:
        n = int(mask.numel() * sparsity)
        if n <= 0:
            return
        with torch.no_grad():
            flat = param.detach().abs().flatten()
            ids = torch.topk(flat, k=n, largest=False).indices
            mask.flatten()[ids] = False
            param.data.flatten()[ids] = 0.0

    def entries(self, prefix: str) -> list[SparseEntry]:
        return [
            SparseEntry(prefix + ".w_in", self.w_in, self.mask_in, self.utility_in, self.protected_in, self.appearance_in),
            SparseEntry(prefix + ".w_rec", self.w_rec, self.mask_rec, self.utility_rec, self.protected_rec, self.appearance_rec),
        ]

    def effective(self, param: torch.Tensor, mask: torch.Tensor, probe: bool) -> torch.Tensor:
        # Probe gives dormant connections a tiny differentiable path only during structural scoring.
        m = mask.to(param.dtype)
        if probe:
            m = m + (~mask).to(param.dtype) * 0.02
        return fake_quant_int8(param) * m

    def step(self, x: torch.Tensor, mem: torch.Tensor, syn: torch.Tensor, prev_spike: torch.Tensor,
             probe: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wi = self.effective(self.w_in, self.mask_in, probe)
        wr = self.effective(self.w_rec, self.mask_rec, probe)
        cur = F.linear(x, wi) + F.linear(prev_spike, wr)
        syn = fake_quant_int8(self.syn_decay * syn + cur, clip=self.mem_clip * 2.0)
        pre = self.mem_decay * mem + syn
        sp = spike_fn(pre - self.threshold)
        mem = fake_quant_int8(pre - sp * self.threshold, clip=self.mem_clip)
        return mem, syn, sp


class OpenGrowthRsnnCode(nn.Module):
    def __init__(self, emb_dim: int, hidden: int, layers: int, initial_sparsity: float,
                 mem_decay: float = 0.92, syn_decay: float = 0.70, threshold: float = 1.0,
                 mem_clip: float = 4.0) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.n_layers = layers
        self.embedding = nn.Embedding(VOCAB, emb_dim)
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(SparseLIFLayer(emb_dim if i == 0 else hidden, hidden, initial_sparsity,
                                              mem_decay, syn_decay, threshold, mem_clip))
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, VOCAB, bias=True)

    def sparse_entries(self) -> list[SparseEntry]:
        out: list[SparseEntry] = []
        for i, layer in enumerate(self.layers):
            out.extend(layer.entries(f"layers.{i}"))
        return out

    def forward(self, ids: torch.Tensor, probe: bool = False) -> torch.Tensor:
        # ids: [B,T]
        b, t = ids.shape
        emb = fake_quant_int8(self.embedding(ids), clip=4.0)
        mem = [torch.zeros(b, self.hidden, device=ids.device) for _ in self.layers]
        syn = [torch.zeros_like(mem[0]) for _ in self.layers]
        spk = [torch.zeros_like(mem[0]) for _ in self.layers]
        outputs = []
        for ti in range(t):
            x = emb[:, ti]
            for li, layer in enumerate(self.layers):
                mem[li], syn[li], spk[li] = layer.step(x, mem[li], syn[li], spk[li], probe=probe)
                x = spk[li]
            # Membrane carries richer recurrent state while remaining fake-INT8 quantized.
            y = self.norm(mem[-1])
            y = fake_quant_int8(y, clip=4.0)
            outputs.append(self.head(y))
        return torch.stack(outputs, dim=1)


class RealPromptDataset:
    PREFIX = b"### Instruction:\n"
    MID = b"\n### Python:\n"
    END = b"\n### End\n"

    def __init__(self, path: Path, seq_len: int, max_instruction_bytes: int = 180) -> None:
        self.seq_len = seq_len
        self.rows: list[tuple[bytes, bytes]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                inst = row["instruction"].encode("utf-8")[:max_instruction_bytes]
                code = row["code"].encode("utf-8")
                self.rows.append((inst, code))
        if not self.rows:
            raise RuntimeError(f"empty dataset: {path}")

    def encoded(self, index: int) -> tuple[list[int], int]:
        inst, code = self.rows[index % len(self.rows)]
        prompt = self.PREFIX + inst + self.MID
        raw = prompt + code + self.END
        ids = [BOS] + list(raw) + [EOS]
        code_start = 1 + len(prompt)
        if len(ids) > self.seq_len + 1:
            ids = ids[: self.seq_len + 1]
        return ids, min(code_start, len(ids) - 1)

    def sample_batch(self, batch_size: int, rng: random.Random, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.full((batch_size, self.seq_len), PAD, dtype=torch.long)
        y = torch.full((batch_size, self.seq_len), -100, dtype=torch.long)
        for bi in range(batch_size):
            ids, code_start = self.encoded(rng.randrange(len(self.rows)))
            n = min(self.seq_len, len(ids) - 1)
            x[bi, :n] = torch.tensor(ids[:n], dtype=torch.long)
            targets = torch.tensor(ids[1:n+1], dtype=torch.long)
            # Only score code/end generation, not copying the prompt header.
            start = max(0, code_start - 1)
            if start < n:
                y[bi, start:n] = targets[start:n]
        return x.to(device), y.to(device), (y != -100).to(device)


class StructuralController:
    def __init__(self, model: OpenGrowthRsnnCode, utility_beta: float = 0.98,
                 grow_fraction: float = 0.005, protect_fraction: float = 0.02,
                 prune_fraction: float = 0.02, important_fraction: float = 0.02,
                 novelty_limit: float = 0.01, stable_cycles: int = 3, regrow_scale: float = 0.03) -> None:
        self.model = model
        self.utility_beta = utility_beta
        self.grow_fraction = grow_fraction
        self.protect_fraction = protect_fraction
        self.prune_fraction = prune_fraction
        self.important_fraction = important_fraction
        self.novelty_limit = novelty_limit
        self.stable_cycles = stable_cycles
        self.regrow_scale = regrow_scale
        self.phase = "growth"
        self.cycle = 0
        self.growth_streak = 0
        self.selection_streak = 0
        self.topology_stable = False
        self.prev_important: set[int] = set()

    def update_utility(self) -> None:
        with torch.no_grad():
            for e in self.model.sparse_entries():
                if e.param.grad is None:
                    continue
                contrib = (e.param.detach() * e.param.grad.detach()).abs()
                e.utility.mul_(self.utility_beta).add_(contrib * (1.0 - self.utility_beta))
                e.utility.mul_(e.mask)

    def _flat(self, attr: str) -> torch.Tensor:
        vals = []
        for e in self.model.sparse_entries():
            v = getattr(e, attr)
            vals.append(v.flatten())
        return torch.cat(vals)

    def _set_global(self, ids: torch.Tensor, mode: str, optimizer: torch.optim.Optimizer) -> None:
        if ids.numel() == 0:
            return
        ids = ids.sort().values.cpu()
        pos = 0
        offset = 0
        entries = self.model.sparse_entries()
        for e in entries:
            n = e.mask.numel()
            lo = int(torch.searchsorted(ids, torch.tensor(offset), right=False))
            hi = int(torch.searchsorted(ids, torch.tensor(offset + n), right=False))
            if hi <= lo:
                offset += n
                continue
            local = ids[lo:hi] - offset
            mf = e.mask.flatten(); pf = e.protected.flatten(); uf = e.utility.flatten(); af = e.appearance.flatten(); wf = e.param.data.flatten()
            if mode == "grow":
                mf[local] = True
                pf[local] = False
                uf[local] = 0
                af[local] = 0
                wf[local] = torch.randn(local.numel(), device=wf.device, dtype=wf.dtype) * self.regrow_scale
            elif mode == "prune":
                mf[local] = False
                pf[local] = False
                uf[local] = 0
                wf[local] = 0
            elif mode == "protect":
                pf[local] = True
            state = optimizer.state.get(e.param, {})
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in state and torch.is_tensor(state[key]) and state[key].numel() == e.param.numel():
                    state[key].flatten()[local.to(state[key].device)] = 0
            offset += n

    def structural_step(self, probe_scores: torch.Tensor, optimizer: torch.optim.Optimizer) -> dict:
        self.cycle += 1
        active = self._flat("mask").bool()
        utility = self._flat("utility")
        protected = self._flat("protected").bool()
        total = active.numel()
        dormant = ~active
        grown = 0
        pruned = 0
        novelty = 1.0

        if self.topology_stable:
            return self.metrics(grown, pruned, novelty)

        if self.phase == "growth":
            candidates = torch.where(dormant)[0]
            if candidates.numel() > 0:
                k = max(1, int(total * self.grow_fraction))
                k = min(k, candidates.numel())
                cscore = probe_scores[candidates]
                topv, topi = torch.topk(cscore, k=k, largest=True)
                active_util = utility[active]
                threshold = active_util.median() if active_util.numel() else torch.tensor(0.0)
                choose = candidates[topi[topv >= threshold * 0.8]]
                self._set_global(choose, "grow", optimizer)
                grown = choose.numel()
            novelty = grown / max(1, total)
            self.growth_streak = self.growth_streak + 1 if novelty < self.novelty_limit else 0
            if self.growth_streak >= self.stable_cycles:
                self.phase = "selection"
            return self.metrics(grown, pruned, novelty)

        # Selection: rolling important set, protect current best 2%, prune current worst 2%, regrow useful dormant links.
        active = self._flat("mask").bool()
        utility = self._flat("utility")
        score = utility + 0.25 * probe_scores.to(utility.device)
        active_ids = torch.where(active)[0]
        if active_ids.numel() == 0:
            return self.metrics(0, 0, 1.0)

        important_k = max(1, int(active_ids.numel() * self.important_fraction))
        important_ids = active_ids[torch.topk(score[active_ids], k=important_k).indices]
        current_important = set(int(x) for x in important_ids.cpu().tolist())
        fresh = len(current_important - self.prev_important)
        novelty = fresh / max(1, important_k)
        self.prev_important = current_important
        self.selection_streak = self.selection_streak + 1 if novelty < self.novelty_limit else 0
        if self.selection_streak >= self.stable_cycles:
            self.phase = "final"
            self.topology_stable = True
            return self.metrics(0, 0, novelty)

        for e in self.model.sparse_entries():
            e.protected.zero_()
        protect_k = max(1, int(active_ids.numel() * self.protect_fraction))
        protect_ids = active_ids[torch.topk(score[active_ids], k=protect_k).indices]
        self._set_global(protect_ids, "protect", optimizer)
        protected = self._flat("protected").bool()
        removable = torch.where(active & ~protected)[0]
        prune_k = min(removable.numel(), max(1, int(active_ids.numel() * self.prune_fraction)))
        if prune_k:
            prune_ids = removable[torch.topk(score[removable], k=prune_k, largest=False).indices]
            self._set_global(prune_ids, "prune", optimizer)
            pruned = prune_ids.numel()

        active2 = self._flat("mask").bool()
        dormant_ids = torch.where(~active2)[0]
        regrow_k = min(pruned, dormant_ids.numel())
        if regrow_k:
            grow_ids = dormant_ids[torch.topk(probe_scores[dormant_ids], k=regrow_k).indices]
            self._set_global(grow_ids, "grow", optimizer)
            grown = grow_ids.numel()
        return self.metrics(grown, pruned, novelty)

    def metrics(self, grown: int, pruned: int, novelty: float) -> dict:
        active = int(self._flat("mask").sum().item())
        protected = int(self._flat("protected").sum().item())
        total = int(self._flat("mask").numel())
        return {
            "cycle": self.cycle, "phase": self.phase, "grown": int(grown), "pruned": int(pruned),
            "active": active, "dormant": total - active, "protected": protected,
            "novelty": float(novelty), "topology_stable": self.topology_stable,
        }


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1), ignore_index=-100)


@torch.no_grad()
def evaluate(model: nn.Module, ds: RealPromptDataset, batches: int, batch_size: int,
             rng: random.Random, device: torch.device) -> float:
    model.eval()
    vals = []
    for _ in range(batches):
        x, y, _ = ds.sample_batch(batch_size, rng, device)
        vals.append(float(loss_fn(model(x), y).item()))
    model.train()
    return sum(vals) / len(vals)


def probe_dormant_scores(model: OpenGrowthRsnnCode, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    logits = model(x, probe=True)
    loss = loss_fn(logits, y)
    loss.backward()
    scores = []
    for e in model.sparse_entries():
        if e.param.grad is None:
            scores.append(torch.zeros_like(e.param).flatten())
        else:
            # Compensate for the 0.02 dormant probe multiplier.
            g = e.param.grad.detach().abs()
            s = torch.where(e.mask, e.utility, g / 0.02)
            scores.append(s.flatten())
    model.zero_grad(set_to_none=True)
    return torch.cat(scores)


def quantize_array(t: torch.Tensor) -> tuple[np.ndarray, float]:
    a = t.detach().cpu().numpy().astype(np.float32)
    mx = float(np.max(np.abs(a))) if a.size else 0.0
    scale = mx / 127.0 if mx > 1e-12 else 1.0
    q = np.clip(np.rint(a / scale), -127, 127).astype(np.int8)
    return q, scale


def export_int8(model: OpenGrowthRsnnCode, out_path: Path, meta: dict) -> None:
    arrays: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    sparse_by_name = {e.name: e for e in model.sparse_entries()}
    for name, p in model.named_parameters():
        source = p.detach()
        if name in sparse_by_name:
            e = sparse_by_name[name]
            source = source * e.mask
            arrays[name + ".mask"] = e.mask.detach().cpu().numpy().astype(np.uint8)
        q, s = quantize_array(source)
        arrays[name] = q
        scales[name] = s
    meta = dict(meta)
    meta["weight_scales"] = scales
    arrays["metadata_json"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(out_path, **arrays)


@torch.no_grad()
def generate(model: OpenGrowthRsnnCode, instruction: str, max_new: int, device: torch.device) -> str:
    model.eval()
    prompt = RealPromptDataset.PREFIX + instruction.encode("utf-8")[:180] + RealPromptDataset.MID
    ids = [BOS] + list(prompt)
    for _ in range(max_new):
        ctx = ids[-256:]
        x = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)[:, -1, :BYTE_VOCAB]
        nxt = int(torch.argmax(logits, dim=-1).item())
        ids.append(nxt)
        if nxt == EOS:
            break
    raw = bytes([i for i in ids[1:] if 0 <= i < 256])
    text = raw.decode("utf-8", errors="replace")
    marker = "### Python:\n"
    return text.split(marker, 1)[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="code_model/data/train.jsonl")
    ap.add_argument("--valid", default="code_model/data/valid.jsonl")
    ap.add_argument("--out-dir", default="code_model/output")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--initial-sparsity", type=float, default=0.75)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--structural-interval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, (os_cpu := __import__('os').cpu_count() or 2))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device} torch={torch.__version__} threads={torch.get_num_threads()}", flush=True)

    train_ds = RealPromptDataset(Path(args.train), args.seq_len)
    valid_ds = RealPromptDataset(Path(args.valid), args.seq_len)
    print(f"[train] real examples train={len(train_ds.rows)} valid={len(valid_ds.rows)}", flush=True)

    model = OpenGrowthRsnnCode(args.emb_dim, args.hidden, args.layers, args.initial_sparsity).to(device)
    controller = StructuralController(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    rng = random.Random(args.seed)
    vrng = random.Random(args.seed + 1)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    initial_val = evaluate(model, valid_ds, batches=4, batch_size=args.batch_size, rng=vrng, device=device)
    best_val = initial_val
    history = []
    structural = []
    start = time.time()
    model.train()

    for step in range(1, args.steps + 1):
        x, y, _ = train_ds.sample_batch(args.batch_size, rng, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        controller.update_utility()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()

        if step == 1 or step % 25 == 0:
            elapsed = time.time() - start
            row = {"step": step, "loss": float(loss.item()), "grad_norm": grad_norm, "seconds": elapsed}
            history.append(row)
            print(f"[train] step={step:04d} loss={row['loss']:.4f} grad={grad_norm:.3f} elapsed={elapsed:.1f}s", flush=True)

        if step % args.structural_interval == 0:
            px, py, _ = train_ds.sample_batch(max(2, args.batch_size // 2), rng, device)
            probe = probe_dormant_scores(model, px, py)
            event = controller.structural_step(probe, optimizer)
            structural.append(event)
            print("[structure] " + json.dumps(event), flush=True)

        if step % 100 == 0 or step == args.steps:
            val = evaluate(model, valid_ds, batches=4, batch_size=args.batch_size, rng=vrng, device=device)
            best_val = min(best_val, val)
            print(f"[eval] step={step} val_loss={val:.4f} best={best_val:.4f}", flush=True)
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "config": vars(args), "controller": controller.metrics(0, 0, 0.0),
            }, out / "checkpoint_latest.pt")

    final_val = evaluate(model, valid_ds, batches=8, batch_size=args.batch_size, rng=vrng, device=device)
    sample = generate(model, "Return the sum of two numbers.", 80, device)
    active = int(sum(e.mask.sum().item() for e in model.sparse_entries()))
    sparse_total = int(sum(e.mask.numel() for e in model.sparse_entries()))
    total_params = sum(p.numel() for p in model.parameters())
    report = {
        "model": "Open-Growth RSNN Code V1",
        "training": "real-source-only",
        "device": str(device),
        "steps": args.steps,
        "initial_val_loss": initial_val,
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "loss_improvement": initial_val - final_val,
        "total_parameters": total_params,
        "sparse_parameters": sparse_total,
        "active_sparse_parameters": active,
        "active_sparse_fraction": active / max(1, sparse_total),
        "full_int8_target": True,
        "qat": "FP32 master weights + fake INT8 weights/activations/states; INT8 deployment export",
        "structural_events": structural,
        "history": history,
        "sample_prompt": "Return the sum of two numbers.",
        "sample_output": sample,
        "elapsed_seconds": time.time() - start,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    export_int8(model, out / "open_growth_rsnn_code_v1_int8.npz", {
        "name": "Open-Growth RSNN Code V1",
        "vocab": VOCAB,
        "tokenizer": "UTF-8 byte-level + BOS/EOS/PAD",
        "emb_dim": args.emb_dim, "hidden": args.hidden, "layers": args.layers,
        "mem_decay": 0.92, "syn_decay": 0.70, "threshold": 1.0, "mem_clip": 4.0,
        "accumulator": "INT32 required for integer deployment MACs",
        "source": "real permissively licensed GitHub repositories; provenance.json accompanies artifact",
    })
    print("[done] " + json.dumps({k: report[k] for k in ("initial_val_loss", "final_val_loss", "loss_improvement", "total_parameters", "active_sparse_fraction", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
