"""
A3 baseline: simple, cache-UNAWARE agent memory (MemGPT/Mem0-style).
Stores conversation facts as embeddings, retrieves top-k most similar on
every turn. No notion of what is already "warm" in the KV cache.
"""
import faiss
import numpy as np


class SimpleMemory:
    def __init__(self, embedder, dim: int = 384):
        self.embedder = embedder
        self.items: list[str] = []
        self.index = faiss.IndexFlatL2(dim)

    def write(self, text: str):
        vec = self.embedder.encode([text]).astype("float32")
        self.index.add(vec)
        self.items.append(text)

    def read(self, query: str, k: int = 5) -> list[str]:
        if not self.items:
            return []
        qvec = self.embedder.encode([query]).astype("float32")
        k = min(k, len(self.items))
        _, idx = self.index.search(qvec, k)
        return [self.items[i] for i in idx[0] if 0 <= i < len(self.items)]

    def __len__(self):
        return len(self.items)
