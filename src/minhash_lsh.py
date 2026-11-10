"""Deterministic MinHash + banded LSH candidate retrieval, followed by exact Jaccard reranking."""
import hashlib
import random
from collections import defaultdict
from similarity import load_movies, jaccard

PRIME = 4294967311

def stable_hash(token):
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big")

def coefficients(num_perm=80, seed=42):
    rng = random.Random(seed)
    return [(rng.randrange(1, PRIME), rng.randrange(PRIME)) for _ in range(num_perm)]

def signature(features, coeffs):
    values = [stable_hash(t) for t in features]
    return tuple(min((a * v + b) % PRIME for v in values) if values else PRIME for a, b in coeffs)

def recommend_lsh(title, top_k=5, num_perm=80, rows_per_band=2):
    if num_perm % rows_per_band:
        raise ValueError("num_perm must be divisible by rows_per_band")
    df = load_movies()
    matches = df[df["title"].str.casefold() == title.casefold()]
    if matches.empty:
        matches = df[df["title"].str.contains(title, case=False, regex=False, na=False)]
    if matches.empty:
        raise ValueError(f"Movie not found: {title}")
    target_idx = matches.index[0]
    coeffs = coefficients(num_perm)
    signatures = {idx: signature(row["feature_set"], coeffs) for idx, row in df.iterrows()}
    buckets = defaultdict(set)
    for idx, sig in signatures.items():
        if not df.at[idx, "feature_set"]:
            continue
        for band in range(num_perm // rows_per_band):
            key = (band, sig[band * rows_per_band:(band + 1) * rows_per_band])
            buckets[key].add(idx)
    target_sig = signatures[target_idx]
    candidates = set()
    for band in range(num_perm // rows_per_band):
        key = (band, target_sig[band * rows_per_band:(band + 1) * rows_per_band])
        candidates.update(buckets.get(key, set()))
    candidates.discard(target_idx)
    target = df.at[target_idx, "feature_set"]
    ranked = sorted(((jaccard(target, df.at[i, "feature_set"]), df.at[i, "title"]) for i in candidates), key=lambda x: (-x[0], x[1]))
    return ranked[:top_k], len(candidates)

if __name__ == "__main__":
    results, count = recommend_lsh("Interstellar")
    print("LSH candidates:", count)
    for score, title in results:
        print(f"{title}: {score:.3f}")
