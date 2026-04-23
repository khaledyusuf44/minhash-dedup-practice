"""Phase 3: shingling + exact Jaccard similarity.

Converts each document into a *set* of contiguous k-grams (word or char), then
computes exact Jaccard similarity on those sets. This is the ground-truth
similarity metric that MinHash will approximate in Phase 4.

Default configuration: word-3-grams on `normalize(text, strip_punct=False)` —
matches the conservative normalization used for exact-dedup in Phase 2, so the
dedup pipeline has a single consistent notion of "same text".

Math recap:
  S_k(D) = { D[i : i+k] : 0 <= i <= |D| - k }       (k-shingle set)
  J(A, B) = |A ∩ B| / |A ∪ B|                       (Jaccard similarity)
  J ∈ [0, 1]; J = 1 iff A = B; J = 0 iff A ∩ B = ∅.

All-pairs Jaccard at N=95,876 is ~4.6 x 10^9 pair comparisons — too expensive
to enumerate. This phase computes Jaccard only on (a) small hand-crafted sanity
pairs, (b) ground-truth pairs, (c) a random-pair baseline. MinHash + LSH in
Phases 4-5 are the answer to the all-pairs problem.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Iterable

# Reuse Phase 2's normalize
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import normalize  # noqa: E402


# ---------- core ----------

def word_shingles(tokens: list[str], k: int) -> set[str]:
    if len(tokens) < k:
        return set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def char_shingles(text: str, k: int) -> set[str]:
    if len(text) < k:
        return set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def shingle(text: str, k: int = 3, mode: str = "word", strip_punct: bool = False) -> set[str]:
    t = normalize(text, strip_punct=strip_punct)
    if mode == "word":
        return word_shingles(t.split(), k)
    if mode == "char":
        return char_shingles(t, k)
    raise ValueError(f"unknown mode: {mode!r}")


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


# ---------- validation helpers ----------

def hand_validation() -> None:
    """Show shingles on 5 tiny texts so boundary behavior is inspectable."""
    print("=== hand-validation: shingles ===")
    cases = [
        ("word k=3 normal", "the model learns from data", 3, "word"),
        ("word k=2 normal", "the model learns from data", 2, "word"),
        ("word k=3 too short", "only two words", 3, "word"),   # 3 tokens, exactly 1 shingle
        ("char k=4", "abcabcabc", 4, "char"),
        ("char k=3 short", "ab", 3, "char"),                    # empty
    ]
    for label, text, k, mode in cases:
        s = shingle(text, k=k, mode=mode)
        print(f"  [{label}]  text={text!r}  k={k}  mode={mode}")
        print(f"    -> {sorted(s)}  (n={len(s)})")
    print()


def sanity_pairs() -> None:
    """Show Jaccard on pairs with known expected behavior."""
    print("=== sanity Jaccard (word 3-grams) ===")
    base = "the quick brown fox jumps over the lazy dog"
    cases = [
        ("exact",          base, base),
        ("1-word sub",     base, "the slow brown fox jumps over the lazy dog"),
        ("2-word sub",     base, "the slow brown hare jumps over the lazy dog"),
        ("prefix added",   base, "breaking news the quick brown fox jumps over the lazy dog"),
        ("reordered",      base, "the lazy dog jumps over the quick brown fox"),
        ("unrelated",      base, "my cat likes to sleep on the warm windowsill"),
    ]
    print(f"  base text: {base!r}")
    for label, a, b in cases:
        sa = shingle(a, k=3, mode="word")
        sb = shingle(b, k=3, mode="word")
        j = jaccard(sa, sb)
        print(f"  {label:15s}  J={j:.3f}  |  |A|={len(sa)}  |B|={len(sb)}  |A∩B|={len(sa & sb)}  |A∪B|={len(sa | sb)}")
    print()


# ---------- corpus-level stats ----------

def ground_truth_jaccard(
    docs_by_id: dict[str, str],
    gt_path: Path,
    k: int,
    mode: str,
) -> tuple[dict[str, list[float]], dict[str, int]]:
    """For each GT pair, compute J if both docs survived Phase 2.
    Returns (jaccards_by_edit_type, missing_by_edit_type)."""
    by_type: dict[str, list[float]] = {}
    missing: dict[str, int] = {}
    with gt_path.open() as f:
        for row in csv.DictReader(f):
            t = row["edit_type"]
            a = docs_by_id.get(row["doc_id_1"])
            b = docs_by_id.get(row["doc_id_2"])
            if a is None or b is None:
                missing[t] = missing.get(t, 0) + 1
                continue
            sa = shingle(a, k=k, mode=mode)
            sb = shingle(b, k=k, mode=mode)
            by_type.setdefault(t, []).append(jaccard(sa, sb))
    return by_type, missing


def random_pair_jaccard(
    docs_by_id: dict[str, str],
    n_pairs: int,
    k: int,
    mode: str,
    seed: int = 0,
) -> list[float]:
    rng = random.Random(seed)
    ids = list(docs_by_id.keys())
    out = []
    for _ in range(n_pairs):
        a, b = rng.sample(ids, 2)
        sa = shingle(docs_by_id[a], k=k, mode=mode)
        sb = shingle(docs_by_id[b], k=k, mode=mode)
        out.append(jaccard(sa, sb))
    return out


def corpus_shingle_counts(
    docs_by_id: dict[str, str],
    k: int,
    mode: str,
    progress_every: int = 20000,
) -> list[int]:
    counts = []
    for i, text in enumerate(docs_by_id.values()):
        counts.append(len(shingle(text, k=k, mode=mode)))
        if (i + 1) % progress_every == 0:
            print(f"  shingled {i+1}/{len(docs_by_id)}")
    return counts


def describe(values: Iterable[float]) -> dict:
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "min": vals[0],
        "p25": vals[n // 4],
        "median": vals[n // 2],
        "p75": vals[3 * n // 4],
        "max": vals[-1],
        "mean": sum(vals) / n,
    }


def _fmt_floats(d: dict) -> str:
    return "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


# ---------- main ----------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    p.add_argument("--input", default=str(root / "data/documents_100k_after_exact.jsonl"))
    p.add_argument("--ground-truth", default=str(root / "data/ground_truth_pairs_100k.csv"))
    p.add_argument("--outputs-dir", default=str(root / "outputs"))
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--mode", default="word", choices=["word", "char"])
    p.add_argument("--n-random-pairs", type=int, default=1000)
    p.add_argument("--skip-corpus-stats", action="store_true")
    args = p.parse_args()

    # Always-run checks first, so bugs surface even without I/O.
    hand_validation()
    sanity_pairs()

    # Load survivors.
    records = [json.loads(line) for line in open(args.input)]
    docs_by_id: dict[str, str] = {r["doc_id"]: r["text"] for r in records}
    print(f"[shingling] loaded {len(docs_by_id)} survivor docs from {args.input}")

    # Ground-truth Jaccard by edit type.
    print(f"\n=== ground-truth Jaccard distribution  (k={args.k} {args.mode}-shingles) ===")
    by_type, missing = ground_truth_jaccard(docs_by_id, Path(args.ground_truth), args.k, args.mode)
    for t in sorted(set(by_type) | set(missing)):
        summary = describe(by_type.get(t, []))
        miss = missing.get(t, 0)
        print(f"  {t:15s}  {_fmt_floats(summary)}  |  removed_by_phase2={miss}")

    # Random-pair baseline (should hug zero).
    print(f"\n=== random-pair Jaccard baseline ({args.n_random_pairs} pairs) ===")
    rps = random_pair_jaccard(docs_by_id, args.n_random_pairs, args.k, args.mode)
    rps_stats = describe(rps)
    rps_stats["fraction_nonzero"] = sum(1 for j in rps if j > 0) / len(rps)
    print(f"  {_fmt_floats(rps_stats)}")

    # Corpus-wide shingle counts (lets Phase 4 budget work).
    corpus_stats: dict
    if args.skip_corpus_stats:
        print("\n[shingling] --skip-corpus-stats set; omitting corpus shingle stats")
        corpus_stats = {}
    else:
        print(f"\n=== corpus shingle counts (k={args.k} {args.mode}) ===")
        counts = corpus_shingle_counts(docs_by_id, args.k, args.mode)
        corpus_stats = describe(counts)
        corpus_stats["total_shingles_across_corpus"] = sum(counts)
        for k, v in corpus_stats.items():
            print(f"  {k}: {v}")

    # Persist.
    report = {
        "k": args.k,
        "mode": args.mode,
        "n_docs": len(docs_by_id),
        "ground_truth_jaccard_by_edit_type": {t: describe(vs) for t, vs in by_type.items()},
        "ground_truth_removed_by_phase2": missing,
        "random_pair_jaccard": rps_stats,
        "corpus_shingle_stats": corpus_stats,
    }
    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = outputs_dir / f"shingle_stats_k{args.k}_{args.mode}.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[shingling] wrote report -> {out_path}")


if __name__ == "__main__":
    main()
