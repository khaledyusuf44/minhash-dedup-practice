"""Phase 4: MinHash from scratch.

For each document D we compute a length-k integer signature sig(D) such that

    Pr[sig_i(A) = sig_i(B)] = J(A, B)     for each coordinate i

where J is exact Jaccard on the shingle sets. The pointwise estimator

    Ĵ(A, B) = (1/k) * Σ_i [sig_i(A) = sig_i(B)]

is unbiased with Var[Ĵ] = J(1−J) / k. At k = 64 and J = 0.5 the one-sigma band
is √(0.25/64) ≈ 6.25 percentage points — the target error range for Phase 4
validation.

Why arithmetic details matter here
----------------------------------
Universal hash family:  h_i(x) = (a_i · x + b_i) mod p

To keep the multiply-then-mod inside numpy's uint64 range we bound every
factor under 2^31:

    p  = 2^31 − 1          (Mersenne prime)
    a  ∈ [1, p)            (uint32 sized)
    b  ∈ [0, p)            (uint32 sized)
    shingle-id ∈ [0, p)    (31-bit truncation of blake2b)

Then a · id fits in 2^62, and +b keeps us inside 2^62. No overflow, no object
dtype, everything vectorizes.

Birthday cost of 31-bit shingle IDs: with ~2M distinct shingles in the corpus
we expect ~10^3 accidental collisions across the whole universe, shifting Ĵ by
noise well under 1 pp per pair. Acceptable for a learning project and still
comfortably below the k=64 estimator's own standard error.

Why min
-------
The min is permutation-invariant, so it doesn't care about ties in the hash;
and for a uniformly random permutation π on the shingle universe,

    Pr[min π(A) = min π(B)] = |A ∩ B| / |A ∪ B| = J(A, B).

We simulate a random permutation with each h_i, so k independent h_i give k
independent Bernoulli(J) draws of the "matching-min" event.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shingling import shingle  # noqa: E402


P = np.uint64((1 << 31) - 1)   # 2,147,483,647 — Mersenne prime
UINT_MAX = np.iinfo(np.uint64).max


def shingle_to_int(s: str) -> int:
    """Stable 31-bit hash of a shingle string (blake2b → 4 bytes → 31-bit mask)."""
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "big") & 0x7FFFFFFF


def make_hashes(k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Draw k hash-function coefficients (a, b) from a fixed seed."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, int(P), size=k, dtype=np.int64).astype(np.uint64)
    b = rng.integers(0, int(P), size=k, dtype=np.int64).astype(np.uint64)
    return a, b


def signature(
    shingle_strings,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Compute the MinHash signature of a single document.

    Empty shingle set -> all-max signature so that (sig_A == sig_B) is False
    whenever either side is empty (empty docs can never match a non-empty doc
    by estimated-J = 0).
    """
    k = a.shape[0]
    if not shingle_strings:
        return np.full(k, UINT_MAX, dtype=np.uint64)
    ids = np.fromiter(
        (shingle_to_int(s) for s in shingle_strings),
        dtype=np.uint64,
        count=len(shingle_strings),
    )
    # (k, n) = (k, 1) * (1, n) + (k, 1), all < P so a*id fits in 2^62.
    hashed = (a[:, None] * ids[None, :] + b[:, None]) % P
    return hashed.min(axis=1).astype(np.uint64)


def signatures_for_corpus(
    docs_by_id: dict[str, str],
    k: int = 64,
    seed: int = 0,
    shingle_k: int = 3,
    shingle_mode: str = "word",
    strip_punct: bool = False,
    progress_every: int = 20000,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Build the N × k signature matrix for a corpus."""
    a, b = make_hashes(k, seed=seed)
    n = len(docs_by_id)
    sigs = np.empty((n, k), dtype=np.uint64)
    doc_ids: list[str] = []

    t0 = time.time()
    for i, (doc_id, text) in enumerate(docs_by_id.items()):
        sh = shingle(text, k=shingle_k, mode=shingle_mode, strip_punct=strip_punct)
        sigs[i] = signature(list(sh), a, b)
        doc_ids.append(doc_id)
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"  signatures: {i+1}/{n}  ({rate:,.0f} docs/s, eta {eta:.1f}s)")
    print(f"  signatures: {n}/{n}  in {time.time() - t0:.2f}s")
    return sigs, doc_ids, a, b


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Agreement ratio across the k coordinates."""
    return float((sig_a == sig_b).mean())


# ----- validation helpers -----

def bernoulli_pair(true_j: float, k: int) -> float:
    """Theoretical std of Ĵ given true J and k hash functions."""
    return float(np.sqrt(true_j * (1.0 - true_j) / k))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    p.add_argument("--input", default=str(root / "data/documents_100k_after_exact.jsonl"))
    p.add_argument("--outputs-dir", default=str(root / "outputs"))
    p.add_argument("--ground-truth", default=str(root / "data/ground_truth_pairs_100k.csv"))
    p.add_argument("--k", type=int, default=64, help="number of hash functions")
    p.add_argument("--shingle-k", type=int, default=3)
    p.add_argument("--shingle-mode", default="word")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-random-pairs", type=int, default=1000)
    args = p.parse_args()

    # 0. Sanity on the universal hash — quick unit-check before corpus work.
    print("=== unit check: universal hash family ===")
    a, b = make_hashes(args.k, seed=args.seed)
    print(f"  a.dtype={a.dtype}  a[:3]={a[:3]}")
    print(f"  b.dtype={b.dtype}  b[:3]={b[:3]}")
    print(f"  all a in [1, P): {bool(((1 <= a) & (a < P)).all())}")
    print(f"  all b in [0, P): {bool((b < P).all())}")

    # 1. Tiny end-to-end check — two identical docs should produce identical sigs.
    s1 = signature(["the quick brown", "quick brown fox"], a, b)
    s2 = signature(["the quick brown", "quick brown fox"], a, b)
    assert np.array_equal(s1, s2), "identical shingle sets must produce identical signatures"
    s3 = signature(["unrelated one", "unrelated two"], a, b)
    agree = estimate_jaccard(s1, s3)
    print(f"  sanity: disjoint-sets Ĵ = {agree:.3f}  (expected ~0.0 ± {bernoulli_pair(0.0, args.k):.3f})")
    print()

    # 2. Load corpus, compute full signature matrix.
    records = [json.loads(line) for line in open(args.input)]
    docs_by_id = {r["doc_id"]: r["text"] for r in records}
    print(f"[minhash] loaded {len(docs_by_id)} survivor docs from {args.input}")
    print(f"[minhash] computing signatures  (k={args.k}, shingle k={args.shingle_k}/{args.shingle_mode})")

    sigs, doc_ids, a, b = signatures_for_corpus(
        docs_by_id,
        k=args.k,
        seed=args.seed,
        shingle_k=args.shingle_k,
        shingle_mode=args.shingle_mode,
    )

    # 3. Persist the matrix + metadata.
    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    sigs_path = outputs_dir / "minhash_signatures.npy"
    map_path = outputs_dir / "doc_id_to_row.csv"
    coef_path = outputs_dir / "minhash_coefficients.npz"

    np.save(sigs_path, sigs)
    with map_path.open("w") as f:
        f.write("doc_id,row_idx\n")
        for row, did in enumerate(doc_ids):
            f.write(f"{did},{row}\n")
    np.savez(coef_path, a=a, b=b, p=np.uint64(P), k=np.int32(args.k),
             shingle_k=np.int32(args.shingle_k),
             shingle_mode=args.shingle_mode)

    print(f"[minhash] signatures -> {sigs_path}  shape={sigs.shape}  dtype={sigs.dtype}")
    print(f"[minhash] row mapping -> {map_path}")
    print(f"[minhash] coefficients -> {coef_path}")

    # 4. Validate on ground-truth pairs.
    import csv
    id_to_row = {did: i for i, did in enumerate(doc_ids)}

    # Recompute exact Jaccard on the fly; bucket by edit_type.
    from shingling import jaccard
    by_type: dict[str, list[tuple[float, float]]] = {}
    with open(args.ground_truth) as f:
        for row in csv.DictReader(f):
            a_id, b_id, t = row["doc_id_1"], row["doc_id_2"], row["edit_type"]
            if a_id not in id_to_row or b_id not in id_to_row:
                continue
            ra, rb = id_to_row[a_id], id_to_row[b_id]
            sa = shingle(docs_by_id[a_id], k=args.shingle_k, mode=args.shingle_mode)
            sb = shingle(docs_by_id[b_id], k=args.shingle_k, mode=args.shingle_mode)
            j_true = jaccard(sa, sb)
            j_hat = estimate_jaccard(sigs[ra], sigs[rb])
            by_type.setdefault(t, []).append((j_true, j_hat))

    print(f"\n=== MinHash estimate vs true Jaccard  (k={args.k}) ===")
    print("  edit_type        n    true_J  Ĵ      err_mean  err_std  theoretical_std")
    report: dict = {"k": args.k, "by_edit_type": {}}
    for t in sorted(by_type):
        pairs = by_type[t]
        trues = np.array([p[0] for p in pairs])
        ests = np.array([p[1] for p in pairs])
        err = ests - trues
        theo = float(np.sqrt(trues * (1 - trues) / args.k).mean())
        print(
            f"  {t:15s}  {len(pairs):5d}  "
            f"{trues.mean():.3f}  {ests.mean():.3f}  "
            f"{err.mean():+.4f}  {err.std():.4f}   {theo:.4f}"
        )
        report["by_edit_type"][t] = {
            "n": len(pairs),
            "mean_true_j": float(trues.mean()),
            "mean_est_j": float(ests.mean()),
            "err_mean": float(err.mean()),
            "err_std": float(err.std()),
            "theoretical_std": theo,
        }

    # 5. Random-pair baseline.
    rng = np.random.default_rng(args.seed + 1)
    all_ids = list(doc_ids)
    rand_pairs = [(rng.choice(all_ids), rng.choice(all_ids)) for _ in range(args.n_random_pairs)]
    rand_pairs = [(a, b) for a, b in rand_pairs if a != b][: args.n_random_pairs]
    rand_trues = []
    rand_ests = []
    for a_id, b_id in rand_pairs:
        sa = shingle(docs_by_id[a_id], k=args.shingle_k, mode=args.shingle_mode)
        sb = shingle(docs_by_id[b_id], k=args.shingle_k, mode=args.shingle_mode)
        rand_trues.append(jaccard(sa, sb))
        rand_ests.append(estimate_jaccard(sigs[id_to_row[a_id]], sigs[id_to_row[b_id]]))
    rand_trues = np.array(rand_trues)
    rand_ests = np.array(rand_ests)
    err = rand_ests - rand_trues
    print(f"\n=== random-pair baseline  (n={len(rand_pairs)}) ===")
    print(
        f"  true_J={rand_trues.mean():.4f}  "
        f"Ĵ={rand_ests.mean():.4f}  "
        f"err_mean={err.mean():+.4f}  err_std={err.std():.4f}"
    )
    report["random_pairs"] = {
        "n": len(rand_pairs),
        "mean_true_j": float(rand_trues.mean()),
        "mean_est_j": float(rand_ests.mean()),
        "err_mean": float(err.mean()),
        "err_std": float(err.std()),
    }

    val_path = outputs_dir / "minhash_validation.json"
    with val_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[minhash] validation report -> {val_path}")


if __name__ == "__main__":
    main()
