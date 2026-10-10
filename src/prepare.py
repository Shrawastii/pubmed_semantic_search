import re
import pandas as pd

df = pd.read_parquet("data/abstracts.parquet")
n0 = len(df)

# collapse extra whitespace
df["text"] = df["text"].apply(lambda t: re.sub(r"\s+", " ", t).strip())

# drop duplicates and empties
df = df.drop_duplicates("pmid").drop_duplicates("text")
df = df[df["text"].str.len() > 0].reset_index(drop=True)

df.to_parquet("data/corpus.parquet", index=False)

print(f"Started with {n0}, kept {len(df)} documents")
print(df["n_words"].describe()[["min", "50%", "max"]])
print(f"Abstracts over 200 words: {(df['n_words'] > 200).mean():.0%}")