#!/usr/bin/env python3
"""Convert the exact stage-1 Open-Growth RSNN PyTorch checkpoint to the compact phone seed format."""
from __future__ import annotations
import argparse, base64, gzip, io, struct
from pathlib import Path
import numpy as np
import torch

MAGIC = 0x4F475231
VERSION = 1

def be_f32(t: torch.Tensor) -> bytes:
    return t.detach().cpu().numpy().astype('>f4', copy=False).tobytes(order='C')

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ck.get('variant') != 'open_growth':
        raise SystemExit(f"wrong variant: {ck.get('variant')}")
    ms = ck['model_state']
    opt = ck['optimizer_state']['state']
    buf = io.BytesIO()
    w = buf.write
    w(struct.pack('>I', MAGIC)); w(struct.pack('>I', VERSION))
    w(struct.pack('>I', 160)); w(struct.pack('>I', 25))
    w(struct.pack('>f', 0.88)); w(struct.pack('>f', 1.0))
    step = int(float(opt[0]['step']))
    w(struct.pack('>q', step)); w(struct.pack('>q', int(ck['cycle']))); w(struct.pack('>q', int(ck['examples_seen'])))
    w(struct.pack('>d', float(ck['completed_training_seconds']))); w(struct.pack('>d', float(ck['best_mae']))); w(struct.pack('>q', int(ck['best_cycle'])))
    for name in ('W_in','W_rec','W_out'): w(be_f32(ms[name]))
    for name in ('M_in','M_rec','M_out'): w((ms[name].detach().cpu().numpy() > 0.5).astype(np.uint8).tobytes(order='C'))
    for pid in (0,1,2): w(be_f32(opt[pid]['exp_avg']))
    for pid in (0,1,2): w(be_f32(opt[pid]['exp_avg_sq']))
    raw = buf.getvalue()
    encoded = base64.b64encode(gzip.compress(raw, compresslevel=9))
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_bytes(encoded)
    print(f"PHONE_SEED_READY raw={len(raw)} encoded={len(encoded)} step={step} cycle={ck['cycle']} examples={ck['examples_seen']}")

if __name__ == '__main__': main()
