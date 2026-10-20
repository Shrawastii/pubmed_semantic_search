import random
from pathlib import Path
import pandas as pd
from src.search import SemanticSearcher, BM25Searcher
from src.config import SEED

OUT = Path("data/label_pool.csv")
if OUT.exists():
    raise SystemExit(f"{OUT} already exists. Delete it only if you want to redo your labels.")

queries = [q.strip() for q in open("data/queries.txt", encoding="utf-8") if q.strip()]
systems = [BM25Searcher(), SemanticSearcher()]
docs = systems[0].df.set_index("pmid")
rng = random.Random(SEED)

rows = []
for q in queries:
    pmids = {r["pmid"] for s in systems for r in s.search(q, k=10)}
    pmids = sorted(pmids)
    rng.shuffle(pmids)
    for p in pmids:
        rows.append({"query": q, "pmid": p,
                     "title": docs.loc[p, "title"],
                     "abstract": docs.loc[p, "abstract"][:700],
                     "relevant": ""})

pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"{len(queries)} queries, {len(rows)} papers to label -> {OUT}")