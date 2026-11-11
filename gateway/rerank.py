"""Cross-encoder reranker on CPU: ONNX Runtime + a Hugging Face tokenizer, no torch.

Default model: Alibaba-NLP/gte-reranker-modernbert-base (149M, Apache-2.0, 8k
context), int8 ONNX export. Fetch it once with `python fetch_reranker.py`.

Everything is imported lazily so a gateway without the files still starts; it
simply answers 404 on /rerank and the caller falls back to fusion scores.
"""

from __future__ import annotations

import logging
import math
import os
import threading

log = logging.getLogger("gateway.rerank")

DEFAULT_REPO = "Alibaba-NLP/gte-reranker-modernbert-base"
DEFAULT_ONNX = "onnx/model_int8.onnx"


def model_dir() -> str:
    return os.path.expanduser(os.environ.get("RERANK_MODEL_DIR", "~/models/gte-reranker-modernbert-base"))


def onnx_file() -> str:
    return os.environ.get("RERANK_ONNX_FILE", DEFAULT_ONNX)


def wanted() -> bool:
    return os.environ.get("RERANK_ENABLED", "1").strip() != "0"


def files_present() -> bool:
    folder = model_dir()
    return os.path.isfile(os.path.join(folder, onnx_file())) and os.path.isfile(
        os.path.join(folder, "tokenizer.json")
    )


def enabled() -> bool:
    return wanted() and files_present()


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


class Reranker:
    def __init__(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        folder = model_dir()
        max_tokens = int(os.environ.get("RERANK_MAX_TOKENS", "512"))
        self.batch = max(1, int(os.environ.get("RERANK_BATCH", "8")))
        self.tokenizer = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_tokens)
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        # Leave the cores to the LLM: reranking 24 short passages needs very little.
        options.intra_op_num_threads = max(1, int(os.environ.get("RERANK_THREADS", "2")))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            os.path.join(folder, onnx_file()), options, providers=["CPUExecutionProvider"]
        )
        self.inputs = {item.name for item in self.session.get_inputs()}

    def scores(self, query: str, documents: list[str]) -> list[float]:
        import numpy as np

        out: list[float] = []
        for start in range(0, len(documents), self.batch):
            pairs = [(query, doc) for doc in documents[start : start + self.batch]]
            encoded = self.tokenizer.encode_batch(pairs)
            feed = {
                "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
            }
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)
            feed = {key: value for key, value in feed.items() if key in self.inputs}
            logits = self.session.run(None, feed)[0]
            out.extend(_sigmoid(float(row[0] if getattr(row, "shape", ()) else row)) for row in logits)
        return out


_instance: Reranker | None = None
_lock = threading.Lock()


def get() -> Reranker:
    global _instance
    with _lock:
        if _instance is None:
            _instance = Reranker()
            log.info("Reranker loaded from %s", model_dir())
        return _instance


def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
    if not documents:
        return []
    scores = get().scores(query, documents)
    ranked = sorted(
        ({"index": index, "relevance_score": score} for index, score in enumerate(scores)),
        key=lambda item: item["relevance_score"],
        reverse=True,
    )
    return ranked[:top_n] if top_n else ranked
