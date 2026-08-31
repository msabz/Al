#!/usr/bin/env python3
"""Thin Kaggle entrypoint for the reviewed MAI5 v3 DeepMind-only trainer."""

import pathlib
import re
import urllib.request

SOURCE_COMMIT = "__SOURCE_COMMIT__"
if not re.fullmatch(r"[0-9a-fA-F]{40}", SOURCE_COMMIT):
    raise RuntimeError(f"SOURCE_COMMIT injection invalid: {SOURCE_COMMIT!r}")

url = f"https://raw.githubusercontent.com/msabz/Al/{SOURCE_COMMIT}/kaggle/deepmind_only_train_v3.py"
target = pathlib.Path("/kaggle/working/_deepmind_only_train_v3.py")
urllib.request.urlretrieve(url, target)
text = target.read_text()
exec(
    compile(text, str(target), "exec"),
    {
        "__name__": "__main__",
        "__file__": str(target),
        "SOURCE_COMMIT": SOURCE_COMMIT,
    },
)
