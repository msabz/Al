#!/usr/bin/env python3
"""Load a trained RSNN Code checkpoint and run deterministic Python generation probes."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import torch

# Import applies the optimized forward + integer-friendly normalization patch.
import code_model.run_integer_only_v2  # noqa: F401
import code_model.train_rsnn_code_v1 as trainer

PROMPTS = [
    "Return the sum of two numbers.",
    "Write a Python function that reverses a string.",
    "Write a function that returns True when a number is even.",
    "Read a UTF-8 text file and return its contents.",
    "Create a Flask route that returns JSON with status ok.",
    "Write a function that finds the maximum value in a list.",
    "Sort a list of dictionaries by the key named age.",
    "Write a class with a method that increments an integer counter.",
]


def clean_generated(text: str) -> str:
    text = text.split("### End", 1)[0]
    return text.replace("\x00", "").strip()


def syntax_ok(code: str) -> tuple[bool, str | None]:
    if not code:
        return False, "empty"
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"{e.msg} at line {e.lineno}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new", type=int, default=220)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = trainer.OpenGrowthRsnnCode(
        emb_dim=int(cfg["emb_dim"]),
        hidden=int(cfg["hidden"]),
        layers=int(cfg["layers"]),
        initial_sparsity=float(cfg["initial_sparsity"]),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    rows = []
    syntax_count = 0
    keyword_count = 0
    keywords = ("def ", "class ", "return", "import ", "from ", "for ", "if ")
    for prompt in PROMPTS:
        raw = trainer.generate(model, prompt, args.max_new, torch.device("cpu"))
        code = clean_generated(raw)
        ok, err = syntax_ok(code)
        if ok:
            syntax_count += 1
        has_keyword = any(k in code for k in keywords)
        if has_keyword:
            keyword_count += 1
        rows.append({
            "prompt": prompt,
            "code": code,
            "syntax_ok": ok,
            "syntax_error": err,
            "has_python_keyword": has_keyword,
        })
        print("\nPROMPT:", prompt)
        print("OUTPUT:\n" + code[:1000])
        print("syntax_ok=", ok, "has_python_keyword=", has_keyword)

    report = {
        "checkpoint_step": int(ckpt.get("step", 0)),
        "config": cfg,
        "prompts": len(rows),
        "syntax_valid": syntax_count,
        "syntax_rate": syntax_count / len(rows),
        "python_keyword_outputs": keyword_count,
        "python_keyword_rate": keyword_count / len(rows),
        "results": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nGENERATION_REPORT", json.dumps({k: report[k] for k in ("prompts", "syntax_valid", "syntax_rate", "python_keyword_outputs", "python_keyword_rate")}, indent=2))


if __name__ == "__main__":
    main()
