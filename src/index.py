import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_parquet("data/corpus.parquet")
emb = np.load("data/embeddings.npy").astype("float32")

index = faiss.IndexFlatIP(emb.shape[1])   # exact search, inner product
index.add(emb)
faiss.write_index(index, "data/index.faiss")
print("Indexed", index.ntotal, "documents")

# quick test: a plain-language query with no exact medical terms
model = SentenceTransformer("all-MiniLM-L6-v2")
q = model.encode(["young woman with psychiatric symptoms and seizures after a viral illness"],
                 normalize_embeddings=True).astype("float32")
scores, ids = index.search(q, 5)
for s, i in zip(scores[0], ids[0]):
    print(f"{s:.3f}  {df.loc[i, 'pmid']}  {df.loc[i, 'title']}")
