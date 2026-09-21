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
