#!/usr/bin/env python3
"""Thin Kaggle entrypoint for the reviewed DeepMind-only trainer."""

import pathlib
import urllib.request

SOURCE_COMMIT = "__SOURCE_COMMIT__"
if not SOURCE_COMMIT or SOURCE_COMMIT == "__SOURCE_COMMIT__":
    raise RuntimeError("SOURCE_COMMIT was not injected by GitHub Actions")

url = f"https://raw.githubusercontent.com/msabz/Al/{SOURCE_COMMIT}/kaggle/deepmind_only_train.py"
target = pathlib.Path("/kaggle/working/_deepmind_only_train.py")
urllib.request.urlretrieve(url, target)
text = target.read_text().replace("__SOURCE_COMMIT__", SOURCE_COMMIT)
target.write_text(text)
exec(compile(text, str(target), "exec"), {"__name__": "__main__", "__file__": str(target)})
