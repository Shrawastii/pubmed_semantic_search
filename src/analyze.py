import numpy as np
import pandas as pd
from src.search import SemanticSearcher, BM25Searcher

qrels = pd.read_csv("eval/qrels_handwritten.csv", dtype={"pmid": str})
rel = qrels.groupby("query")["pmid"].apply(set).to_dict()


def ndcg(ranked, relevant):
    dcg = sum(1 / np.log2(i + 2) for i, p in enumerate(ranked[:10]) if p in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), 10)))
    return dcg / idcg


bm25, sem = BM25Searcher(), SemanticSearcher()
rows = []
for q, rp in rel.items():
    b = ndcg([r["pmid"] for r in bm25.search(q, 10)], rp)
    s = ndcg([r["pmid"] for r in sem.search(q, 10)], rp)
    rows.append({"query": q, "bm25": round(b, 3), "semantic": round(s, 3),
                 "diff": round(s - b, 3)})

df = pd.DataFrame(rows).sort_values("diff")
df.to_csv("results/per_query.csv", index=False)

tol = 0.02
print(f"Semantic won: {(df['diff'] > tol).sum()}, "
      f"BM25 won: {(df['diff'] < -tol).sum()}, "
      f"tied (within {tol}): {(df['diff'].abs() <= tol).sum()}")
pd.set_option("display.max_colwidth", 70)
print("\nBM25 did best relative to semantic:")
print(df.head(5).to_string(index=False))
print("\nSemantic did best relative to BM25:")
print(df.tail(5).to_string(index=False))
print("\nBoth struggled (lowest combined):")
df["both"] = df["bm25"] + df["semantic"]
print(df.nsmallest(5, "both")[["query", "bm25", "semantic"]].to_string(index=False))