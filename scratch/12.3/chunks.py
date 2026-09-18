import time
from resolve.kb.index import get_client, load_config
from resolve.guards.injection import score_text, threshold
cfg = load_config()
client = get_client()
resp = client.search(index=cfg.get("index", "resolve-kb"), body={"query": {"match_all": {}}, "size": 1000, "_source": ["chunk_id", "space", "text"]})
chunks = [h["_source"] for h in resp["hits"]["hits"]]
t0 = time.perf_counter()
flagged = [(c["chunk_id"], c["space"], score_text(c["text"])) for c in chunks if score_text(c["text"])["risk"] >= threshold()]
print(f"{len(flagged)}/{len(chunks)} chunks at or above {threshold()}, {time.perf_counter()-t0:.0f} s")
for f in flagged: print("  ", f)
