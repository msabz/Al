#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path, old, new):
    p = ROOT / path
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one target, found {count}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1))
    print("patched", path)

# SymPy replacement must be simultaneous. Sequential substitution can collapse
# two source variables when one source name is also x/y, e.g. a->x then x->y.
replace_once(
    "colab/train_v5_deepmind.py",
    "        out.append(str(e.lhs.subs(repl)).replace('**','^')+'='+str(e.rhs.subs(repl)).replace('**','^'))",
    "        out.append(str(e.lhs.subs(repl, simultaneous=True)).replace('**','^')+'='+str(e.rhs.subs(repl, simultaneous=True)).replace('**','^'))",
)

# The pre-generated dataset adapter has the same collision risk because it used
# two successive regex substitutions. Replace all source variable tokens in one pass.
replace_once(
    "colab/turbo_train_v5.py",
    "def rename_var(text, old, new):\n    return re.sub(rf\"\\b{re.escape(old)}\\b\", new, text)\n",
    "def rename_var(text, old, new):\n"
    "    return re.sub(rf\"\\b{re.escape(old)}\\b\", new, text)\n\n"
    "def rename_vars_simultaneous(text, mapping):\n"
    "    if not mapping: return text\n"
    "    pattern = r\"\\b(?:\" + \"|\".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r\")\\b\"\n"
    "    return re.sub(pattern, lambda m: mapping[m.group(0)], text)\n",
)
replace_once(
    "colab/turbo_train_v5.py",
    "    renamed = []\n    for eq in parts:\n        for old, new in repl.items():\n            eq = rename_var(eq, old, new)\n        renamed.append(eq)",
    "    renamed = [rename_vars_simultaneous(eq, repl) for eq in parts]",
)

print("DEEPMIND_COLLISION_SAFE_RENAMING_OK")
