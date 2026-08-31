import json
import math
import os
import random
import time
from pathlib import Path

import torch
from torch import nn

SEED = 20260830
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

print("=== MathAI Kaggle mechanism smoke test ===", flush=True)
print(f"device={device}", flush=True)
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

# Tiny learned calculator. Inputs are normalized [a, b, op], where op=+1 for + and -1 for -.
# A small MLP must learn how the operator changes the contribution of b.
class TinyAddSub(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def make_batch(n: int, low: int, high: int):
    a = torch.randint(low, high + 1, (n, 1), device=device).float()
    b = torch.randint(low, high + 1, (n, 1), device=device).float()
    op = torch.where(
        torch.rand((n, 1), device=device) < 0.5,
        torch.ones((n, 1), device=device),
        -torch.ones((n, 1), device=device),
    )
    scale = float(max(abs(low), abs(high), 1))
    x = torch.cat((a / scale, b / scale, op), dim=1)
    y = (a + op * b) / (2.0 * scale)
    return x, y, scale


def evaluate(model, low, high, n=10000):
    model.eval()
    with torch.no_grad():
        x, y, scale = make_batch(n, low, high)
        pred = model(x)
        pred_real = pred * (2.0 * scale)
        y_real = y * (2.0 * scale)
        err = pred_real - y_real
        mae = err.abs().mean().item()
        rmse = torch.sqrt((err * err).mean()).item()
        exact_025 = (err.abs() <= 0.25).float().mean().item()
        exact_050 = (err.abs() <= 0.50).float().mean().item()
    return {
        "range": [low, high],
        "samples": n,
        "mae": mae,
        "rmse": rmse,
        "within_0_25": exact_025,
        "within_0_50": exact_050,
    }


model = TinyAddSub().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()

baseline = evaluate(model, -100, 100, 4000)
print("baseline=" + json.dumps(baseline, sort_keys=True), flush=True)

started = time.time()
steps = 2500
batch_size = 1024
for step in range(1, steps + 1):
    model.train()
    x, y, _ = make_batch(batch_size, -100, 100)
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step == 1 or step % 100 == 0:
        print(f"step={step}/{steps} loss={loss.item():.8f}", flush=True)

# In-distribution holdout and a wider range to demonstrate whether the learned rule extrapolates at all.
holdout = evaluate(model, -100, 100, 20000)
wider = evaluate(model, -150, 150, 10000)

cases = [
    (2, 3, "+"),
    (9, 4, "-"),
    (-7, 12, "+"),
    (25, -8, "-"),
    (100, 100, "+"),
    (-100, 100, "-"),
]
examples = []
model.eval()
with torch.no_grad():
    for a, b, symbol in cases:
        op = 1.0 if symbol == "+" else -1.0
        x = torch.tensor([[a / 100.0, b / 100.0, op]], device=device)
        predicted = model(x).item() * 200.0
        expected = a + (b if symbol == "+" else -b)
        examples.append({
            "expression": f"{a}{symbol}{b}",
            "expected": expected,
            "predicted": predicted,
            "abs_error": abs(predicted - expected),
        })

improvement = baseline["rmse"] / max(holdout["rmse"], 1e-12)
# This is intentionally strict enough to prove training happened but not so strict that minor GPU variation causes false failure.
passed = (
    holdout["rmse"] < 0.35
    and holdout["within_0_50"] > 0.98
    and improvement > 50.0
    and max(x["abs_error"] for x in examples) < 0.75
)

report = {
    "test": "kaggle_add_sub_training_mechanism",
    "source_commit": "__SOURCE_COMMIT__",
    "seed": SEED,
    "device": str(device),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "architecture": "3-32-32-1 tanh MLP",
    "train_range": [-100, 100],
    "steps": steps,
    "batch_size": batch_size,
    "baseline": baseline,
    "holdout": holdout,
    "wider_range": wider,
    "rmse_improvement_factor": improvement,
    "examples": examples,
    "elapsed_seconds": time.time() - started,
    "verdict": "PASS" if passed else "FAIL",
}

model_path = OUT / "add_sub_smoke_model.pt"
report_path = OUT / "add_sub_smoke_report.json"
evidence_path = OUT / "ADD_SUB_SMOKE_EVIDENCE.txt"

torch.save({
    "state_dict": model.cpu().state_dict(),
    "architecture": "3-32-32-1 tanh MLP",
    "input": "[a/100,b/100,+1 for add or -1 for subtract]",
    "output": "result/200",
    "seed": SEED,
}, model_path)
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

evidence_lines = [
    "MathAI Kaggle add/sub mechanism smoke test",
    f"VERDICT={report['verdict']}",
    f"DEVICE={report['device']}",
    f"GPU={report['gpu']}",
    f"BASELINE_RMSE={baseline['rmse']:.6f}",
    f"HOLDOUT_RMSE={holdout['rmse']:.6f}",
    f"HOLDOUT_WITHIN_0.50={holdout['within_0_50']:.6f}",
    f"IMPROVEMENT_FACTOR={improvement:.2f}",
]
for e in examples:
    evidence_lines.append(
        f"CASE {e['expression']} expected={e['expected']} predicted={e['predicted']:.6f} abs_error={e['abs_error']:.6f}"
    )
evidence_path.write_text("\n".join(evidence_lines) + "\n")

print(json.dumps(report, indent=2, sort_keys=True), flush=True)
print(f"MODEL_PATH={model_path} bytes={model_path.stat().st_size}", flush=True)
print(f"REPORT_PATH={report_path}", flush=True)
print(f"EVIDENCE_PATH={evidence_path}", flush=True)
print(f"FINAL_VERDICT={report['verdict']}", flush=True)

if not passed:
    raise SystemExit(2)
