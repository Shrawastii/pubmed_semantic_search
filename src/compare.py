from src.search import SemanticSearcher, BM25Searcher

QUERIES = [
    "young woman with psychiatric symptoms and seizures after a viral illness",
    "how is autoimmune encephalitis diagnosed when antibody tests are negative",
    "brain inflammation caused by mosquito bites",
    "outcomes of herpes encephalitis treated late",
    "anti-NMDAR",
]

systems = [BM25Searcher(), SemanticSearcher()]

for q in QUERIES:
    print("\n" + "=" * 80 + f"\nQUERY: {q}")
    for s in systems:
        print(f"\n  [{s.name}]")
        for r in s.search(q, k=5):
            print(f"   {r['rank']}. ({r['pmid']}) {r['title'][:90]}")