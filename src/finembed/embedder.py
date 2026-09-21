"""OpenAI embeddings with batching and retries."""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

log = logging.getLogger(__name__)


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-large", dimensions: Optional[int] = 1536,
                 batch_size: int = 64, max_retries: int = 6):
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set (put it in .env)")
        self.client = OpenAI()
        self.model, self.dimensions = model, dimensions
        self.batch_size, self.max_retries = batch_size, max_retries
        self.name = f"openai:{model}" + (f":{dimensions}" if dimensions else "")
        self.calls = 0  # API requests made (used to verify re-runs are free)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(texts[i:i + self.batch_size]))
        return out

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        from openai import AuthenticationError, BadRequestError

        kw = {"model": self.model, "input": batch}
        if self.dimensions:
            kw["dimensions"] = self.dimensions
        for attempt in range(self.max_retries):
            try:
                resp = self.client.embeddings.create(**kw)
                self.calls += 1
                return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
            except (AuthenticationError, BadRequestError):
                raise
            except Exception as e:  # rate limits, timeouts, 5xx
                if attempt == self.max_retries - 1:
                    raise
                wait = min(2 ** attempt, 30)
                log.warning("OpenAI embeddings failed (%s); retry %d in %ss", type(e).__name__, attempt + 1, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")


class BGEEmbedder:
    """Local CPU embeddings via transformers (CLS pooling). Spec: BAAI/bge-small-en-v1.5 (384-d).

    Does not import sentence-transformers: this guest image blocks scipy's Cython DLLs.
    """

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", batch_size: int = 16):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model
        self.batch_size = batch_size
        self._torch = torch
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model, local_files_only=True
            )
            self.client = AutoModel.from_pretrained(
                model, local_files_only=True
            )
        except OSError:
            # First-time setup may download once. Subsequent talk/demo starts
            # use the local cache and make no network request.
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.client = AutoModel.from_pretrained(model)
        self.client.eval()
        self.dimensions = int(self.client.config.hidden_size)
        self.name = model
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        torch = self._torch
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            with torch.no_grad():
                hidden = self.client(**encoded).last_hidden_state[:, 0]
                hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
            out.extend(hidden.cpu().numpy().astype("float32").tolist())
            self.calls += 1
            log.info("embedded %d/%d", min(i + self.batch_size, len(texts)), len(texts))
        return out
