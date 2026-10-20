import random
import pandas as pd
from pathlib import Path
from src.search import HybridSearcher
from src.config import SEED

OUT = Path("data/label_pool_extra.csv")
if OUT.exists():
    raise SystemExit(f"{OUT} already exists. Delete it only if you want to redo these labels.")

lab = pd.read_csv("data/label_pool.csv", dtype={"pmid": str}, encoding="utf-8-sig")
seen = set(zip(lab["query"], lab["pmid"]))

h = HybridSearcher()
docs = h.df.set_index("pmid")
rng = random.Random(SEED)

rows, total = [], 0
for q in lab["query"].unique():
    top = [r["pmid"] for r in h.search(q, k=10)]
    new = [p for p in top if (q, p) not in seen]
    total += len(new)
    rng.shuffle(new)
    for p in new:
        rows.append({"query": q, "pmid": p, "title": docs.loc[p, "title"],
                     "abstract": docs.loc[p, "abstract"][:700], "relevant": ""})

print(f"{total} of {lab['query'].nunique() * 10} hybrid top-10 results were unlabeled")
pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"Saved {len(rows)} rows -> {OUT}")