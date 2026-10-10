import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.config import set_seed

set_seed()
df = pd.read_parquet("data/corpus.parquet")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(
    df["text"].tolist(), batch_size=64,
    show_progress_bar=True, normalize_embeddings=True)

np.save("data/embeddings.npy", embeddings)
print("Saved embeddings:", embeddings.shape)