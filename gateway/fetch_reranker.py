"""Download the reranker files once (the Docker build calls this too).

    python fetch_reranker.py [target_dir]

Only the tokenizer and one ONNX file are fetched (int8 by default, ~150 MB).
"""

from __future__ import annotations

import os
import sys
import urllib.request

from rerank import DEFAULT_REPO, model_dir, onnx_file

FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json")


def fetch(target: str, repo: str = DEFAULT_REPO) -> None:
    for name in (*FILES, onnx_file()):
        path = os.path.join(target, name)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = f"https://huggingface.co/{repo}/resolve/main/{name}"
        print(f"fetching {url}")
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)


if __name__ == "__main__":
    fetch(sys.argv[1] if len(sys.argv) > 1 else model_dir(), os.environ.get("RERANK_REPO", DEFAULT_REPO))
