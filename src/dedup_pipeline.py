"""Phase 6: Jaccard verification -> threshold -> union-find -> keep-rule -> deduped corpus.

Inputs
------
  data/documents_100k_after_exact.jsonl   (Phase 2 survivors)
  outputs/candidate_pairs.csv             (Phase 5 LSH candidates)

Pipeline
--------
  1. For each candidate pair, recompute exact Jaccard on full word-3-gram shingle
     sets. This is the decision metric — MinHash/LSH only served as a filter.
  2. Apply threshold tau (default 0.80). Keep pairs with J >= tau.
  3. Build union-find over accepted pairs. Each connected component is a dup
     cluster. Duplicates are a graph, not a list: {A,B,C} with J(A,B)>=tau and
     J(B,C)>=tau is one cluster of size 3, not two separate pair decisions.
  4. Per cluster, apply keep-rule (default: "longest" body wins). State it, don't
     mix rules across clusters.
  5. Emit deduped_documents.jsonl + duplicate_pairs.csv + dedup_summary.json.

Tradeoffs baked in
------------------
  * tau = 0.80 — standard for pretraining (Gopher, RefinedWeb region). Moving to
    0.90 is conservative (only near-exact), 0.70 starts catching paraphrases.
  * keep-rule "longest" biases the surviving distribution toward long-form.
    "first-seen" biases toward input-file ordering. Both are biases; document
    which one is in force.
  * Decision metric is *exact Jaccard*, not Ĵ from MinHash. LSH is a filter;
    the verifier must not be noisy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shingling import jaccard, shingle  # noqa: E402


class UnionFind:
    """Nearly-linear union-find with path compression. Auto-adds nodes on find."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry

    def components(self) -> list[set[str]]:
        groups: dict[str, set[str]] = defaultdict(set)
        for x in list(self.parent):
            groups[self.find(x)].add(x)
        return list(groups.values())


# ---------- IO ----------

def load_candidates(path: Path) -> list[tuple[str, str, int]]:
    out = []
    with path.open() as f:
        next(f)  # header
        for line in f:
            a, b, sb = line.strip().split(",")
            out.append((a, b, int(sb)))
    return out


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------- pipeline stages ----------

def verify_candidates(
    candidates: list[tuple[str, str, int]],
    shingles: dict[str, set],
    tau: float,
) -> tuple[list[tuple], list[tuple[str, str]]]:
    """Returns (all_verified_rows, above_threshold_pairs).

    all_verified_rows: (doc_id_1, doc_id_2, shared_bands, jaccard, decision)
    above_threshold_pairs: (a, b) tuples with J >= tau
    """
    verified = []
    above = []
    for a, b, sb in candidates:
        j = jaccard(shingles[a], shingles[b])
        decision = "duplicate" if j >= tau else "below_threshold"
        verified.append((a, b, sb, j, decision))
        if decision == "duplicate":
            above.append((a, b))
    return verified, above


def build_clusters(pairs: list[tuple[str, str]]) -> list[set[str]]:
    uf = UnionFind()
    for a, b in pairs:
        uf.union(a, b)
    return uf.components()


def select_survivors(
    clusters: list[set[str]],
    docs_by_id: dict[str, dict],
    doc_order: dict[str, int],
    rule: str,
) -> set[str]:
    """Given clusters and a keep rule, return the set of doc_ids to REMOVE."""
    to_remove: set[str] = set()
    for cluster in clusters:
        if rule == "first-seen":
            survivor = min(cluster, key=lambda d: doc_order[d])
        elif rule == "longest":
            # longest text; ties broken by earliest doc_order
            survivor = max(cluster, key=lambda d: (len(docs_by_id[d]["text"]), -doc_order[d]))
        elif rule == "shortest":
            survivor = min(cluster, key=lambda d: (len(docs_by_id[d]["text"]), doc_order[d]))
        else:
            raise ValueError(f"unknown keep-rule: {rule!r}")
        for d in cluster:
            if d != survivor:
                to_remove.add(d)
    return to_remove


# ---------- ground-truth evaluation ----------

def evaluate_ground_truth(
    gt_path: Path,
    final_ids: set[str],
) -> tuple[dict[str, dict[str, int]], int, int]:
    """For each GT pair, classify outcome.
      - caught_one_removed  : exactly one side is in the final corpus.
      - caught_both_removed : neither side survives (both folded into another cluster).
      - missed              : both sides still in final corpus (duplicate survived).

    Returns (by_edit_type, total_caught, total_missed).
    """
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_caught = 0
    total_missed = 0
    with gt_path.open() as f:
        for row in csv.DictReader(f):
            t = row["edit_type"]
            a_in = row["doc_id_1"] in final_ids
            b_in = row["doc_id_2"] in final_ids
            if a_in and b_in:
                by_type[t]["missed"] += 1
                total_missed += 1
            elif a_in or b_in:
                by_type[t]["caught_one_removed"] += 1
                total_caught += 1
            else:
                by_type[t]["caught_both_removed"] += 1
                total_caught += 1
    return {k: dict(v) for k, v in by_type.items()}, total_caught, total_missed


# ---------- main ----------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    p.add_argument("--survivors", default=str(root / "data/documents_100k_after_exact.jsonl"))
    p.add_argument("--candidates", default=str(root / "outputs/candidate_pairs.csv"))
    p.add_argument("--full-corpus", default=str(root / "data/documents_100k.jsonl"))
    p.add_argument("--ground-truth", default=str(root / "data/ground_truth_pairs_100k.csv"))
    p.add_argument("--outputs-dir", default=str(root / "outputs"))
    p.add_argument("--tau", type=float, default=0.80)
    p.add_argument("--keep-rule", choices=["longest", "first-seen", "shortest"], default="longest")
    p.add_argument("--shingle-k", type=int, default=3)
    p.add_argument("--shingle-mode", default="word")
    p.add_argument("--suffix", default=None,
                   help="filename suffix on the output files (default: none at tau=0.80+longest)")
    args = p.parse_args()

    # 1. Load corpus + candidates.
    survivors = load_jsonl(Path(args.survivors))
    docs_by_id = {d["doc_id"]: d for d in survivors}
    doc_order = {d["doc_id"]: i for i, d in enumerate(survivors)}
    candidates = load_candidates(Path(args.candidates))
    print(f"[dedup] Phase-2 survivors: {len(survivors):,}")
    print(f"[dedup] candidate pairs:   {len(candidates):,}")

    # 2. Precompute shingles only for docs that appear in a candidate pair.
    involved = set()
    for a, b, _ in candidates:
        involved.add(a)
        involved.add(b)
    t0 = time.time()
    shingles_by_id: dict[str, set] = {}
    for did in involved:
        shingles_by_id[did] = shingle(
            docs_by_id[did]["text"],
            k=args.shingle_k,
            mode=args.shingle_mode,
        )
    print(f"[dedup] shingled {len(involved):,} candidate docs in {time.time()-t0:.2f}s")

    # 3. Verify.
    t1 = time.time()
    verified, above = verify_candidates(candidates, shingles_by_id, args.tau)
    print(f"[dedup] verified in {time.time()-t1:.2f}s    "
          f"above tau={args.tau}: {len(above):,}    "
          f"below: {len(candidates) - len(above):,}")

    # 4. Cluster.
    clusters = build_clusters(above)
    docs_in_clusters = sum(len(c) for c in clusters)
    print(f"[dedup] clusters: {len(clusters):,}    docs in clusters: {docs_in_clusters:,}")
    if clusters:
        sizes = sorted((len(c) for c in clusters), reverse=True)
        print(f"[dedup] cluster sizes — top 10: {sizes[:10]}  (max={sizes[0]}, median={sizes[len(sizes)//2]})")

    # 5. Pick survivors.
    to_remove = select_survivors(clusters, docs_by_id, doc_order, args.keep_rule)
    print(f"[dedup] keep-rule={args.keep_rule}    removing {len(to_remove):,} docs")

    # 6. Write outputs.
    suffix = args.suffix
    if suffix is None:
        suffix = "" if (args.tau, args.keep_rule) == (0.80, "longest") else f"_tau{args.tau}_{args.keep_rule}"
    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    dup_pairs_path = outputs_dir / f"duplicate_pairs{suffix}.csv"
    with dup_pairs_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id_1", "doc_id_2", "shared_bands", "jaccard", "decision"])
        for a, b, sb, j, dec in verified:
            w.writerow([a, b, sb, f"{j:.4f}", dec])

    final_docs = [d for d in survivors if d["doc_id"] not in to_remove]
    dedup_path = outputs_dir / f"deduped_documents{suffix}.jsonl"
    write_jsonl(dedup_path, final_docs)

    # 7. Summary counts.
    original_size = sum(1 for _ in open(args.full_corpus))
    phase2_removed = original_size - len(survivors)
    phase6_removed = len(to_remove)
    total_removed = phase2_removed + phase6_removed
    summary = {
        "tau": args.tau,
        "keep_rule": args.keep_rule,
        "shingle": f"k={args.shingle_k} {args.shingle_mode}",
        "original_corpus_size": original_size,
        "phase2_exact_dedup_removed": phase2_removed,
        "phase2_survivors": len(survivors),
        "candidate_pairs_evaluated": len(candidates),
        "above_threshold_pairs": len(above),
        "below_threshold_pairs": len(candidates) - len(above),
        "clusters_formed": len(clusters),
        "docs_in_clusters": docs_in_clusters,
        "phase6_near_dedup_removed": phase6_removed,
        "final_corpus_size": len(final_docs),
        "total_removed": total_removed,
        "reduction_pct": 100 * total_removed / original_size,
    }

    # 8. GT recall.
    gt_path = Path(args.ground_truth)
    if gt_path.exists():
        final_ids = {d["doc_id"] for d in final_docs}
        by_type, caught, missed = evaluate_ground_truth(gt_path, final_ids)
        recall = caught / (caught + missed) if (caught + missed) else 0.0
        summary["ground_truth_total_caught"] = caught
        summary["ground_truth_total_missed"] = missed
        summary["ground_truth_overall_recall"] = recall
        summary["ground_truth_by_edit_type"] = by_type

        print("\n=== ground-truth recall (whole pipeline: Phase 2 + Phase 6) ===")
        print("  edit_type        caught  missed  recall")
        for t in sorted(by_type):
            d = by_type[t]
            c = d.get("caught_one_removed", 0) + d.get("caught_both_removed", 0)
            m = d.get("missed", 0)
            r = c / (c + m) if (c + m) else 0.0
            print(f"  {t:15s}  {c:5d}   {m:5d}   {r*100:5.1f}%")
        print(f"  {'OVERALL':15s}  {caught:5d}   {missed:5d}   {recall*100:5.1f}%")

    summary_path = outputs_dir / f"dedup_summary{suffix}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== counts summary ===")
    for k, v in summary.items():
        if k.startswith("ground_truth"):
            continue
        print(f"  {k}: {v}")
    print(f"\n[dedup] deduped corpus  -> {dedup_path}")
    print(f"[dedup] duplicate pairs -> {dup_pairs_path}")
    print(f"[dedup] summary          -> {summary_path}")


if __name__ == "__main__":
    main()
