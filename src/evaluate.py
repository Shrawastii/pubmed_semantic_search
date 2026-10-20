import numpy as np
import pandas as pd
from pathlib import Path
from src.search import SemanticSearcher, BM25Searcher, HybridSearcher
from src.logger import log_result

lab = pd.read_csv("data/label_pool.csv", dtype={"pmid": str}, encoding="utf-8-sig")
extra = Path("data/label_pool_extra.csv")
if extra.exists():
    lab = pd.concat([lab, pd.read_csv(extra, dtype={"pmid": str}, encoding="utf-8-sig")],
                    ignore_index=True)
lab["relevant"] = pd.to_numeric(lab["relevant"], errors="coerce").fillna(0).astype(int)
rel = lab[lab.relevant == 1].groupby("query")["pmid"].apply(set).to_dict()
print(f"{len(rel)} of {lab['query'].nunique()} queries have at least one relevant paper")

# save labels without abstract text, so they can be committed to GitHub
Path("eval").mkdir(exist_ok=True)
lab[lab.relevant == 1][["query", "pmid"]].to_csv("eval/qrels_handwritten.csv", index=False)


def metrics(ranked, relevant):
    out = {}
    for k in (1, 5, 10):
        top = ranked[:k]
        hits = len(set(top) & relevant)
        out[f"hit@{k}"] = float(hits > 0)
        out[f"recall@{k}"] = hits / len(relevant)
    rr = 0.0
    for i, p in enumerate(ranked[:10]):
        if p in relevant:
            rr = 1 / (i + 1)
            break
    out["mrr"] = rr
    dcg = sum(1 / np.log2(i + 2) for i, p in enumerate(ranked[:10]) if p in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), 10)))
    out["ndcg@10"] = dcg / idcg
    return out


def evaluate(system):
    rows = [metrics([r["pmid"] for r in system.search(q, k=10)], rp)
            for q, rp in rel.items()]
    return pd.DataFrame(rows).mean().round(3).to_dict()


results = {}
for s in [BM25Searcher(), SemanticSearcher(), HybridSearcher()]:
    results[s.name] = evaluate(s)
    log_result(s.name, eval_set="handwritten", n_queries=len(rel), **results[s.name])

print(pd.DataFrame(results).T.to_string())