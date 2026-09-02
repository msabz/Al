#!/usr/bin/env python3
"""Collect real prompt->Python examples from permissively licensed public repositories.

No synthetic code is generated. Each instruction is an actual upstream docstring and each
training target is the exact corresponding function/class source from a pinned checkout.
Train and validation repositories are disjoint. Dataset files are generated only during CI.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable

TRAIN_REPOS = [
    ("psf/requests", "Apache-2.0"),
    ("pallets/flask", "BSD-3-Clause"),
    ("encode/httpx", "BSD-3-Clause"),
    ("pytest-dev/pytest", "MIT"),
    ("fastapi/fastapi", "MIT"),
    ("django/django", "BSD-3-Clause"),
    ("sqlalchemy/sqlalchemy", "MIT"),
    ("numpy/numpy", "BSD-3-Clause"),
    ("pallets/click", "BSD-3-Clause"),
    ("Textualize/rich", "MIT"),
    ("encode/starlette", "BSD-3-Clause"),
    ("psf/black", "MIT"),
    ("aio-libs/aiohttp", "Apache-2.0"),
    ("pypa/pip", "MIT"),
    ("tornadoweb/tornado", "Apache-2.0"),
]
VALID_REPOS = [
    ("pydantic/pydantic", "MIT"),
    ("python-attrs/attrs", "MIT"),
    ("pandas-dev/pandas", "BSD-3-Clause"),
]

SKIP_PARTS = {
    ".git", ".github", ".venv", "venv", "build", "dist", "site-packages",
    "node_modules", "docs", "doc", "examples", "example", "tests", "test",
}


def run(cmd: list[str], cwd: Path | None = None) -> str:
    p = subprocess.run(cmd, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.stdout.strip()


def clone(repo: str, dst: Path) -> str:
    url = f"https://github.com/{repo}.git"
    run(["git", "clone", "--depth", "1", "--filter=blob:none", "--quiet", url, str(dst)])
    return run(["git", "rev-parse", "HEAD"], cwd=dst)


def source_segment(text: str, node: ast.AST) -> str | None:
    try:
        return ast.get_source_segment(text, node)
    except Exception:
        return None


def iter_examples(repo_dir: Path, repo: str, commit: str, license_name: str) -> Iterable[dict]:
    for path in sorted(repo_dir.rglob("*.py")):
        rel = path.relative_to(repo_dir)
        if any(part.lower() in SKIP_PARTS for part in rel.parts):
            continue
        try:
            if path.stat().st_size > 400_000:
                continue
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (UnicodeDecodeError, SyntaxError, OSError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            doc = ast.get_docstring(node, clean=True)
            if not doc:
                continue
            doc = " ".join(doc.split())
            if not (20 <= len(doc) <= 900):
                continue
            code = source_segment(text, node)
            if not code:
                continue
            code = code.strip()
            if not (40 <= len(code) <= 6000):
                continue
            if code.count("\n") > 220:
                continue
            digest = hashlib.sha256((doc + "\n" + code).encode()).hexdigest()
            yield {
                "instruction": doc,
                "code": code,
                "repo": repo,
                "commit": commit,
                "path": rel.as_posix(),
                "license": license_name,
                "sha256": digest,
            }


def collect_group(repos: list[tuple[str, str]], out_path: Path, max_per_repo: int, work: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    total = 0
    provenance = []
    with out_path.open("w", encoding="utf-8") as out:
        for repo, license_name in repos:
            dst = work / repo.replace("/", "__")
            print(f"[data] cloning {repo}", flush=True)
            commit = clone(repo, dst)
            count = 0
            for ex in iter_examples(dst, repo, commit, license_name):
                if ex["sha256"] in seen:
                    continue
                seen.add(ex["sha256"])
                out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count += 1
                total += 1
                if count >= max_per_repo:
                    break
            provenance.append({"repo": repo, "commit": commit, "license": license_name, "examples": count})
            print(f"[data] {repo}: {count} examples @ {commit[:12]}", flush=True)
    return {"examples": total, "sources": provenance}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="code_model/data")
    ap.add_argument("--max-per-repo", type=int, default=2200)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="rsnn-real-python-") as td:
        work = Path(td)
        train_meta = collect_group(TRAIN_REPOS, out_dir / "train.jsonl", args.max_per_repo, work)
        valid_meta = collect_group(VALID_REPOS, out_dir / "valid.jsonl", max(1200, args.max_per_repo // 2), work)

    if train_meta["examples"] < 5000:
        raise RuntimeError(f"Too few real training examples: {train_meta['examples']}")
    if valid_meta["examples"] < 1000:
        raise RuntimeError(f"Too few real validation examples: {valid_meta['examples']}")

    meta = {
        "policy": "real-source-only: actual upstream docstrings paired with exact source code; no synthetic examples",
        "split_policy": "repository-disjoint train/validation",
        "train": train_meta,
        "valid": valid_meta,
    }
    (out_dir / "provenance.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
