# pubmed_semantic_search
## Results

Evaluated on 30 hand-labeled queries over 4,023 PubMed encephalitis abstracts (2022-2026).

| System | hit@1 | MRR | nDCG@10 |
|---|---|---|---|
| BM25 | 0.867 | 0.933 | 0.870 |
| Semantic (MiniLM + FAISS) | 0.967 | 0.983 | 0.928 |
| Hybrid (RRF) | 0.967 | 0.983 | 0.889 |

## Error analysis

On 30 hand-labeled queries, semantic search won on 16, BM25 on 8, and 6 were tied
(nDCG@10 difference under 0.02). [Your confirmed observations here.]
Per-query scores are in `results/per_query.csv`.

## Limitations

- Labels were made by one person, by pooling the top 10 from each system (two rounds).
- 30 queries is small, and gaps of a few points are within noise.
- Queries are written in plain language, which favors semantic search.
- MiniLM truncates long abstracts at about 256 word-pieces.
- Research literature search only, not medical advice.