import time
import numpy as np
import requests

queries = [q.strip() for q in open("data/queries.txt", encoding="utf-8") if q.strip()]
URL = "http://127.0.0.1:8000/search"

for q in queries[:5]:                      # warm-up, not counted
    requests.get(URL, params={"q": q})

times = []
for i in range(200):
    q = queries[i % len(queries)]
    t = time.perf_counter()
    requests.get(URL, params={"q": q, "k": 10}).raise_for_status()
    times.append((time.perf_counter() - t) * 1000)

print(f"n=200  p50={np.percentile(times, 50):.1f} ms  "
      f"p95={np.percentile(times, 95):.1f} ms  max={max(times):.1f} ms")