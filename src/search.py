import re
import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CORPUS = "data/corpus.parquet"
MODEL_NAME = "all-MiniLM-L6-v2"

STOP = {"a", "an", "the", "of", "in", "on", "and", "or", "to", "for", "with",
        "is", "are", "was", "were", "by", "as", "at", "from", "that", "this",
        "after", "be", "it", "its", "who", "what", "how", "which", "do", "does"}


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


class BaseSearcher:
    name = "base"

    def __init__(self):
        self.df = pd.read_parquet(CORPUS)

    def _top(self, query, k):
        """Return (row_indices, scores). Implemented by each searcher."""
        raise NotImplementedError

    def search(self, query, k=10):
        idx, scores = self._top(query, k)
        return [
            {"rank": r + 1,
             "pmid": self.df.loc[i, "pmid"],
             "title": self.df.loc[i, "title"],
             "year": self.df.loc[i, "year"],
             "score": float(s)}
            for r, (i, s) in enumerate(zip(idx, scores))
        ]


class SemanticSearcher(BaseSearcher):
    name = "semantic"

    def __init__(self, model_name=MODEL_NAME, index_path="data/index.faiss"):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)

    def _top(self, query, k):
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, k)
        return ids[0], scores[0]


class BM25Searcher(BaseSearcher):
    name = "bm25"

    def __init__(self):
        super().__init__()
        self.bm25 = BM25Okapi([tokenize(t) for t in self.df["text"]])

    def _top(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(-scores)[:k]
        return idx, scores[idx]