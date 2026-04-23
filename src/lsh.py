"""Phase 5: LSH banding + candidate-pair generation.

Splits the MinHash signature matrix into b bands of r rows (b * r = k). Two
documents become 'candidates' if their r-tuple slice is byte-identical in at
least one band.

Math:
    P(band collision | J = s) = s^r
    P(≥1 band collision       | J = s) = 1 - (1 - s^r)^b
    approximate s* threshold        ≈ (1/b)^(1/r)

Tuning table at k = 64:
    b=16 r=4  -> s* ≈ 0.500  (permissive, many candidates, catches moderate dups)
    b=8  r=8  -> s* ≈ 0.760  (strict, fewer candidates, misses J<0.7)
    b=32 r=2  -> s* ≈ 0.177  (very permissive, candidate explosion)

The output candidate_pairs.csv carries (doc_id_1, doc_id_2, shared_bands).
shared_bands ∈ [1, b] is a secondary signal: pairs that collide in many bands
are very likely true duplicates, even before Jaccard verification.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def s_curve_threshold(b: int, r: int) -> float:
    return float((1.0 / b) ** (1.0 / r))


def p_candidate(s: float, b: int, r: int) -> float:
    return float(1 - (1 - s ** r) ** b)


def load_signatures(path: Path) -> np.ndarray:
    return np.load(path)


def load_row_mapping(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    with path.open() as f:
        next(f)  # header
        for line in f:
            did, row = line.strip().split(",")
            mapping[int(row)] = did
    return mapping


def generate_candidates(sigs: np.ndarray, b: int, r: int) -> dict[tuple[int, int], int]:
    """Return {(low_row, high_row): shared_bands_count} across all b bands."""
    k = sigs.shape[1]
    if b * r != k:
        raise ValueError(f"b * r must equal signature length k; got {b*r} != {k}")
    n = sigs.shape[0]

    shared: dict[tuple[int, int], int] = defaultdict(int)
    t0 = time.time()
    for band_idx in range(b):
        start = band_idx * r
        band = np.ascontiguousarray(sigs[:, start : start + r])
        buckets: dict[bytes, list[int]] = defaultdict(list)
        for row_idx in range(n):
            buckets[band[row_idx].tobytes()].append(row_idx)

        multi_buckets = [v for v in buckets.values() if len(v) >= 2]
        new_pairs_this_band = 0
        for docs in multi_buckets:
            # Sort once so pair tuples are always (low, high)
            docs.sort()
            m = len(docs)
            for i in range(m):
                a = docs[i]
                for j in range(i + 1, m):
                    c = docs[j]
                    shared[(a, c)] += 1
                    new_pairs_this_band += 1

        elapsed = time.time() - t0
        print(
            f"  band {band_idx+1:2d}/{b}  "
            f"buckets={len(buckets):7d}  "
            f"multi={len(multi_buckets):5d}  "
            f"pair_events={new_pairs_this_band:7d}  "
            f"unique_pairs={len(shared):7d}  "
            f"elapsed={elapsed:.2f}s"
        )
    return dict(shared)


def write_candidates(
    shared: dict[tuple[int, int], int],
    row_to_id: dict[int, str],
    out_path: Path,
) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id_1", "doc_id_2", "shared_bands"])
        for (a, c), count in shared.items():
            w.writerow([row_to_id[a], row_to_id[c], count])


def cross_check_ground_truth(
    shared: dict[tuple[int, int], int],
    row_to_id: dict[int, str],
    gt_path: Path,
) -> dict:
    id_to_row = {v: k for k, v in row_to_id.items()}
    cand_set = set(shared.keys())
    caught = defaultdict(int)
    missed = defaultdict(int)
    unreachable = defaultdict(int)  # one side removed by Phase 2
    with gt_path.open() as f:
        for row in csv.DictReader(f):
            t = row["edit_type"]
            a_id, b_id = row["doc_id_1"], row["doc_id_2"]
            if a_id not in id_to_row or b_id not in id_to_row:
                unreachable[t] += 1
                continue
            a, c = id_to_row[a_id], id_to_row[b_id]
            pair = (a, c) if a < c else (c, a)
            if pair in cand_set:
                caught[t] += 1
            else:
                missed[t] += 1
    return {
        "caught": dict(caught),
        "missed": dict(missed),
        "unreachable": dict(unreachable),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    p.add_argument("--signatures", default=str(root / "outputs/minhash_signatures.npy"))
    p.add_argument("--row-map", default=str(root / "outputs/doc_id_to_row.csv"))
    p.add_argument("--outputs-dir", default=str(root / "outputs"))
    p.add_argument("--ground-truth", default=str(root / "data/ground_truth_pairs_100k.csv"))
    p.add_argument("-b", "--bands", type=int, default=16)
    p.add_argument("-r", "--rows", type=int, default=4)
    p.add_argument("--suffix", default=None,
                   help="filename suffix (default: none for 16x4, else _b{b}r{r})")
    args = p.parse_args()

    sigs = load_signatures(Path(args.signatures))
    row_to_id = load_row_mapping(Path(args.row_map))
    n = sigs.shape[0]
    k = sigs.shape[1]
    b, r = args.bands, args.rows
    all_pairs = n * (n - 1) // 2

    print(f"[lsh] signatures: N={n:,}  k={k}")
    print(f"[lsh] banding: b={b}  r={r}  (k = b*r = {b*r})")
    s_star = s_curve_threshold(b, r)
    print(f"[lsh] informal threshold  s* ≈ {s_star:.3f}")
    print(
        f"[lsh] P(candidate) by true J:  "
        f"J=0.3→{p_candidate(0.3, b, r):.3f}  "
        f"J=0.5→{p_candidate(0.5, b, r):.3f}  "
        f"J=0.7→{p_candidate(0.7, b, r):.3f}  "
        f"J=0.9→{p_candidate(0.9, b, r):.3f}"
    )
    print()

    shared = generate_candidates(sigs, b, r)
    n_cand = len(shared)

    suffix = args.suffix
    if suffix is None:
        suffix = "" if (b, r) == (16, 4) else f"_b{b}r{r}"
    out_path = Path(args.outputs_dir) / f"candidate_pairs{suffix}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_candidates(shared, row_to_id, out_path)

    counts = sorted(shared.values(), reverse=True)
    stats = {
        "b": b,
        "r": r,
        "k": k,
        "N_docs": n,
        "all_pairs": all_pairs,
        "candidate_pairs": n_cand,
        "reduction_ratio": 1.0 - (n_cand / all_pairs),
        "candidates_as_pct_of_all_pairs": 100 * n_cand / all_pairs,
        "mean_shared_bands": float(np.mean(counts)) if counts else 0.0,
        "max_shared_bands": int(counts[0]) if counts else 0,
        "top_10_shared_bands": counts[:10],
        "s_curve_threshold": s_star,
        "output_file": str(out_path),
    }

    gt_path = Path(args.ground_truth)
    if gt_path.exists():
        cross = cross_check_ground_truth(shared, row_to_id, gt_path)
        stats["ground_truth"] = cross

    stats_path = Path(args.outputs_dir) / f"lsh_stats{suffix}.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[lsh] wrote {n_cand:,} candidate pairs -> {out_path}")
    print(f"[lsh] stats -> {stats_path}")
    print(
        f"[lsh] reduction: candidates are {stats['candidates_as_pct_of_all_pairs']:.4f}% "
        f"of {all_pairs:,} all-pairs  ({stats['reduction_ratio']*100:.4f}% reduction)"
    )
    print(f"[lsh] shared_bands: mean {stats['mean_shared_bands']:.2f}, max {stats['max_shared_bands']}, top10={stats['top_10_shared_bands']}")

    if "ground_truth" in stats:
        print("\n=== LSH recall against ground truth ===")
        print("  edit_type        caught / reachable   (unreachable: removed by Phase 2)")
        gt = stats["ground_truth"]
        for t in sorted(set(gt["caught"]) | set(gt["missed"]) | set(gt["unreachable"])):
            c = gt["caught"].get(t, 0)
            m = gt["missed"].get(t, 0)
            u = gt["unreachable"].get(t, 0)
            reach = c + m
            rate = c / reach if reach else 0.0
            print(f"  {t:15s}  {c:5d} / {reach:5d}  ({rate*100:5.1f}%)    unreachable={u}")


if __name__ == "__main__":
    main()
