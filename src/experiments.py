"""Phase 7: experiments + error analysis.

Runs four sweeps on the dedup pipeline:

  D. threshold       tau ∈ {0.70, 0.80, 0.90}                       (cheapest)
  C. banding         (b, r) ∈ {(16,4), (8,8), (32,2)} at k=64       (reuses sigs)
  B. signature length k ∈ {32, 64, 128}                              (rebuild sigs)
  A. shingle type    word-3 / word-5 / char-5                        (full rebuild)

For each configuration we record precision and recall against the injected
ground truth, plus the counts that explain the result (candidates generated,
pairs above tau, clusters, final corpus size).

After the sweeps, we sample pairs for manual inspection: 20 true-positive
detections, 20 detections that don't match ground truth (potential false
positives or natural duplicates), and 20 ground-truth pairs the pipeline
missed entirely. Output goes to outputs/manual_inspection.md.

All results roll up into outputs/experiments_report.json and a human-readable
summary table printed to stdout.
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shingling import shingle, jaccard  # noqa: E402
from minhash import make_hashes, signature  # noqa: E402
from lsh import generate_candidates  # noqa: E402
from dedup_pipeline import UnionFind, select_survivors  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
SURVIVORS_PATH = ROOT / "data/documents_100k_after_exact.jsonl"
FULL_CORPUS_PATH = ROOT / "data/documents_100k.jsonl"
GT_PATH = ROOT / "data/ground_truth_pairs_100k.csv"
OUTPUTS_DIR = ROOT / "outputs"


# ---------- cached loaders ----------

_docs_by_id: dict[str, dict] | None = None
_doc_order: dict[str, int] | None = None
_gt_pairs: list[dict] | None = None
_shingle_cache: dict[tuple, dict[str, set]] = {}


def docs_by_id() -> dict[str, dict]:
    global _docs_by_id, _doc_order
    if _docs_by_id is None:
        with SURVIVORS_PATH.open() as f:
            rows = [json.loads(line) for line in f]
        _docs_by_id = {r["doc_id"]: r for r in rows}
        _doc_order = {r["doc_id"]: i for i, r in enumerate(rows)}
    return _docs_by_id


def doc_order() -> dict[str, int]:
    docs_by_id()
    assert _doc_order is not None
    return _doc_order


def gt_pairs() -> list[dict]:
    global _gt_pairs
    if _gt_pairs is None:
        with GT_PATH.open() as f:
            _gt_pairs = list(csv.DictReader(f))
    return _gt_pairs


def shingled(shingle_k: int, shingle_mode: str) -> dict[str, set]:
    key = (shingle_k, shingle_mode)
    if key not in _shingle_cache:
        t0 = time.time()
        docs = docs_by_id()
        out: dict[str, set] = {}
        for did, rec in docs.items():
            out[did] = shingle(rec["text"], k=shingle_k, mode=shingle_mode)
        print(f"    shingled corpus  k={shingle_k} mode={shingle_mode}  in {time.time()-t0:.2f}s")
        _shingle_cache[key] = out
    return _shingle_cache[key]


# ---------- pipeline stages ----------

def compute_signatures(
    shingles_by_id: dict[str, set],
    docs_by_id_map: dict[str, dict],
    k_hashes: int,
    seed: int = 0,
) -> tuple[np.ndarray, list[str], float]:
    a, b = make_hashes(k_hashes, seed=seed)
    ids = list(docs_by_id_map.keys())
    sigs = np.empty((len(ids), k_hashes), dtype=np.uint64)
    t0 = time.time()
    for i, did in enumerate(ids):
        sigs[i] = signature(list(shingles_by_id[did]), a, b)
    elapsed = time.time() - t0
    return sigs, ids, elapsed


def run_lsh(sigs: np.ndarray, b: int, r: int) -> tuple[dict[tuple[int, int], int], float]:
    t0 = time.time()
    shared = generate_candidates(sigs, b, r)
    return shared, time.time() - t0


def verify_cluster_decide(
    candidates: list[tuple[str, str, int]],
    shingles_by_id: dict[str, set],
    docs_by_id_map: dict[str, dict],
    doc_order_map: dict[str, int],
    tau: float,
    keep_rule: str = "longest",
) -> tuple[int, set[str], list[tuple[str, str, float]]]:
    """Returns (num_above, to_remove, above_pairs_with_j)."""
    above = []
    for a, b, _ in candidates:
        j = jaccard(shingles_by_id[a], shingles_by_id[b])
        if j >= tau:
            above.append((a, b, j))
    uf = UnionFind()
    for a, b, _ in above:
        uf.union(a, b)
    clusters = uf.components()
    to_remove = select_survivors(clusters, docs_by_id_map, doc_order_map, keep_rule)
    return len(above), to_remove, above


def evaluate(
    final_ids: set[str],
    detected_pairs: set[tuple[str, str]],
    gt: list[dict],
) -> dict:
    """Compute per-edit-type recall + strict precision-against-GT."""
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_caught = 0
    total_missed = 0
    gt_pair_set = set()
    for row in gt:
        a, b = row["doc_id_1"], row["doc_id_2"]
        pair = (a, b) if a < b else (b, a)
        gt_pair_set.add(pair)
        t = row["edit_type"]
        a_in = a in final_ids
        b_in = b in final_ids
        if a_in and b_in:
            by_type[t]["missed"] += 1
            total_missed += 1
        else:
            by_type[t]["caught"] += 1
            total_caught += 1

    detected_in_gt = len(detected_pairs & gt_pair_set)
    detected_not_in_gt = len(detected_pairs) - detected_in_gt
    precision_strict = (detected_in_gt / len(detected_pairs)) if detected_pairs else 0.0

    overall_recall = total_caught / (total_caught + total_missed) if (total_caught + total_missed) else 0.0
    return {
        "overall_recall": overall_recall,
        "precision_strict_vs_gt": precision_strict,
        "detected_total": len(detected_pairs),
        "detected_in_gt": detected_in_gt,
        "detected_not_in_gt": detected_not_in_gt,
        "caught": total_caught,
        "missed": total_missed,
        "by_edit_type": {t: dict(v) for t, v in by_type.items()},
    }


# ---------- one-shot configuration runner ----------

def run_config(
    shingle_k: int,
    shingle_mode: str,
    k_hashes: int,
    b: int,
    r: int,
    tau: float,
    keep_rule: str = "longest",
    label: str = "",
) -> dict:
    docs_map = docs_by_id()
    order = doc_order()

    sh = shingled(shingle_k, shingle_mode)
    sigs, row_ids, sig_time = compute_signatures(sh, docs_map, k_hashes)
    row_to_id = {i: did for i, did in enumerate(row_ids)}

    shared, lsh_time = run_lsh(sigs, b, r)
    # Convert (row, row) pairs to (doc_id, doc_id) tuples.
    candidates_list: list[tuple[str, str, int]] = []
    for (ra, rc), cnt in shared.items():
        candidates_list.append((row_to_id[ra], row_to_id[rc], cnt))
    num_candidates = len(candidates_list)

    t_verify = time.time()
    num_above, to_remove, above = verify_cluster_decide(
        candidates_list, sh, docs_map, order, tau, keep_rule
    )
    verify_time = time.time() - t_verify

    final_ids = {did for did in docs_map if did not in to_remove}
    detected_pairs = {(a, b) if a < b else (b, a) for a, b, _ in above}
    metrics = evaluate(final_ids, detected_pairs, gt_pairs())

    return {
        "label": label,
        "shingle_k": shingle_k,
        "shingle_mode": shingle_mode,
        "k_hashes": k_hashes,
        "b": b,
        "r": r,
        "tau": tau,
        "keep_rule": keep_rule,
        "num_candidates": num_candidates,
        "num_above_tau": num_above,
        "final_corpus_size": len(docs_map) - len(to_remove),
        "sig_time_s": round(sig_time, 2),
        "lsh_time_s": round(lsh_time, 2),
        "verify_time_s": round(verify_time, 2),
        **metrics,
    }


# ---------- experiment D: threshold sweep ----------
# Reuses one set of signatures + candidates. Cheapest.

def exp_D_threshold_sweep() -> list[dict]:
    print("\n========== EXP D — threshold sweep ==========")
    docs_map = docs_by_id()
    order = doc_order()
    sh = shingled(3, "word")
    sigs, row_ids, sig_time = compute_signatures(sh, docs_map, 64)
    row_to_id = {i: did for i, did in enumerate(row_ids)}
    shared, lsh_time = run_lsh(sigs, 16, 4)
    candidates_list = [(row_to_id[ra], row_to_id[rc], cnt) for (ra, rc), cnt in shared.items()]

    out = []
    for tau in [0.70, 0.80, 0.90]:
        num_above, to_remove, above = verify_cluster_decide(candidates_list, sh, docs_map, order, tau)
        final_ids = {did for did in docs_map if did not in to_remove}
        detected = {(a, b) if a < b else (b, a) for a, b, _ in above}
        metrics = evaluate(final_ids, detected, gt_pairs())
        out.append({
            "label": f"D tau={tau}",
            "tau": tau,
            "shingle_k": 3, "shingle_mode": "word", "k_hashes": 64, "b": 16, "r": 4,
            "num_candidates": len(candidates_list),
            "num_above_tau": num_above,
            "final_corpus_size": len(docs_map) - len(to_remove),
            **metrics,
        })
        _print_row(out[-1])
    return out


# ---------- experiment C: banding sweep ----------

def exp_C_banding_sweep() -> list[dict]:
    print("\n========== EXP C — banding sweep (k=64) ==========")
    docs_map = docs_by_id()
    order = doc_order()
    sh = shingled(3, "word")
    sigs, row_ids, _ = compute_signatures(sh, docs_map, 64)
    row_to_id = {i: did for i, did in enumerate(row_ids)}

    out = []
    for b, r in [(16, 4), (8, 8), (32, 2)]:
        shared, lsh_time = run_lsh(sigs, b, r)
        candidates_list = [(row_to_id[ra], row_to_id[rc], cnt) for (ra, rc), cnt in shared.items()]
        num_above, to_remove, above = verify_cluster_decide(candidates_list, sh, docs_map, order, 0.80)
        final_ids = {did for did in docs_map if did not in to_remove}
        detected = {(a, b2) if a < b2 else (b2, a) for a, b2, _ in above}
        metrics = evaluate(final_ids, detected, gt_pairs())
        s_star = (1.0 / b) ** (1.0 / r)
        out.append({
            "label": f"C b={b} r={r}",
            "b": b, "r": r, "shingle_k": 3, "shingle_mode": "word", "k_hashes": 64, "tau": 0.80,
            "s_curve_threshold_approx": round(s_star, 3),
            "num_candidates": len(candidates_list),
            "num_above_tau": num_above,
            "final_corpus_size": len(docs_map) - len(to_remove),
            "lsh_time_s": round(lsh_time, 2),
            **metrics,
        })
        _print_row(out[-1])
    return out


# ---------- experiment B: signature length ----------

def exp_B_signature_length() -> list[dict]:
    print("\n========== EXP B — signature length (word-3, (k/4, 4) banding, tau=0.80) ==========")
    docs_map = docs_by_id()
    order = doc_order()
    sh = shingled(3, "word")
    out = []
    for k_h in [32, 64, 128]:
        b_ = k_h // 4
        r_ = 4
        sigs, row_ids, sig_time = compute_signatures(sh, docs_map, k_h)
        row_to_id = {i: did for i, did in enumerate(row_ids)}
        id_to_row = {did: i for i, did in enumerate(row_ids)}
        shared, lsh_time = run_lsh(sigs, b_, r_)
        candidates_list = [(row_to_id[ra], row_to_id[rc], cnt) for (ra, rc), cnt in shared.items()]
        num_above, to_remove, above = verify_cluster_decide(candidates_list, sh, docs_map, order, 0.80)
        final_ids = {did for did in docs_map if did not in to_remove}
        detected = {(a, b2) if a < b2 else (b2, a) for a, b2, _ in above}
        metrics = evaluate(final_ids, detected, gt_pairs())
        # Estimator sanity: mean/std of (Ĵ - J) on 500 random GT pairs
        rng = random.Random(0)
        gt_sample = [row for row in gt_pairs()
                     if row["doc_id_1"] in docs_map and row["doc_id_2"] in docs_map]
        gt_sample = rng.sample(gt_sample, min(500, len(gt_sample)))
        errs = []
        for row in gt_sample:
            a, b2 = row["doc_id_1"], row["doc_id_2"]
            true_j = jaccard(sh[a], sh[b2])
            ra, rb = id_to_row[a], id_to_row[b2]
            est_j = float((sigs[ra] == sigs[rb]).mean())
            errs.append(est_j - true_j)
        errs_np = np.array(errs)
        memory_mb = sigs.nbytes / 1e6
        out.append({
            "label": f"B k={k_h}",
            "k_hashes": k_h, "b": b_, "r": r_, "tau": 0.80,
            "shingle_k": 3, "shingle_mode": "word",
            "num_candidates": len(candidates_list),
            "num_above_tau": num_above,
            "final_corpus_size": len(docs_map) - len(to_remove),
            "sig_time_s": round(sig_time, 2),
            "sig_memory_mb": round(memory_mb, 1),
            "estimator_err_mean": round(float(errs_np.mean()), 4),
            "estimator_err_std": round(float(errs_np.std()), 4),
            "theoretical_err_std_at_j_0p5": round(float(np.sqrt(0.25 / k_h)), 4),
            **metrics,
        })
        _print_row(out[-1])
    return out


# ---------- experiment A: shingle type ----------

def exp_A_shingle_type() -> list[dict]:
    print("\n========== EXP A — shingle type (k=64, (16,4), tau=0.80) ==========")
    configs = [
        (3, "word"),
        (5, "word"),
        (5, "char"),
    ]
    out = []
    for sk, sm in configs:
        result = run_config(sk, sm, 64, 16, 4, 0.80, label=f"A {sk}-{sm}")
        out.append(result)
        _print_row(result)
    return out


# ---------- manual inspection ----------

def manual_inspection(k_samples: int = 20) -> None:
    """Sample 20 TPs, FPs, FNs at default (word-3, k=64, (16,4), tau=0.80) and
    write outputs/manual_inspection.md."""
    print("\n========== manual inspection (default config) ==========")
    docs_map = docs_by_id()
    order = doc_order()
    sh = shingled(3, "word")
    sigs, row_ids, _ = compute_signatures(sh, docs_map, 64)
    row_to_id = {i: did for i, did in enumerate(row_ids)}
    shared, _ = run_lsh(sigs, 16, 4)
    candidates_list = [(row_to_id[ra], row_to_id[rc], cnt) for (ra, rc), cnt in shared.items()]
    _, to_remove, above = verify_cluster_decide(candidates_list, sh, docs_map, order, 0.80)
    final_ids = {did for did in docs_map if did not in to_remove}

    detected = {((a, b) if a < b else (b, a)): j for a, b, j in above}
    gt_pair_set = {((r["doc_id_1"], r["doc_id_2"]) if r["doc_id_1"] < r["doc_id_2"] else (r["doc_id_2"], r["doc_id_1"])): r["edit_type"] for r in gt_pairs()}

    tp_pairs = [(p, j) for p, j in detected.items() if p in gt_pair_set]
    fp_pairs = [(p, j) for p, j in detected.items() if p not in gt_pair_set]

    # False negatives: GT pairs where both sides still in final corpus.
    fn_pairs = []
    for row in gt_pairs():
        a, b = row["doc_id_1"], row["doc_id_2"]
        if a in final_ids and b in final_ids:
            key = (a, b) if a < b else (b, a)
            fn_pairs.append((key, row["edit_type"]))

    rng = random.Random(7)
    tp_sample = rng.sample(tp_pairs, min(k_samples, len(tp_pairs)))
    fp_sample = rng.sample(fp_pairs, min(k_samples, len(fp_pairs)))
    fn_sample = rng.sample(fn_pairs, min(k_samples, len(fn_pairs)))

    out_path = OUTPUTS_DIR / "manual_inspection.md"
    with out_path.open("w") as f:
        f.write("# Manual inspection — default (word-3, k=64, (16,4), tau=0.80)\n\n")
        f.write(f"Samples: {len(tp_sample)} TP / {len(fp_sample)} FP (not in GT) / {len(fn_sample)} FN.\n\n")

        f.write("## True positives (detected above tau, match ground truth)\n\n")
        for (a, b), j in tp_sample:
            kind = gt_pair_set[(a, b)]
            ta = docs_map[a]["text"][:220]
            tb = docs_map[b]["text"][:220]
            f.write(f"### pair ({a}, {b})  J={j:.3f}  edit_type=`{kind}`\n\n")
            f.write(f"- {a}: {ta}{'...' if len(docs_map[a]['text']) > 220 else ''}\n")
            f.write(f"- {b}: {tb}{'...' if len(docs_map[b]['text']) > 220 else ''}\n\n")

        f.write("## 'Not in ground truth' detections (natural dup or false positive?)\n\n")
        for (a, b), j in fp_sample:
            ta = docs_map[a]["text"][:220]
            tb = docs_map[b]["text"][:220]
            f.write(f"### pair ({a}, {b})  J={j:.3f}\n\n")
            f.write(f"- {a}: {ta}{'...' if len(docs_map[a]['text']) > 220 else ''}\n")
            f.write(f"- {b}: {tb}{'...' if len(docs_map[b]['text']) > 220 else ''}\n\n")

        f.write("## False negatives (ground-truth pairs the pipeline missed)\n\n")
        for (a, b), kind in fn_sample:
            # recompute J at word-3 to see why it missed
            j = jaccard(sh[a], sh[b])
            ta = docs_map[a]["text"][:220]
            tb = docs_map[b]["text"][:220]
            f.write(f"### pair ({a}, {b})  J={j:.3f}  edit_type=`{kind}`\n\n")
            f.write(f"- {a}: {ta}{'...' if len(docs_map[a]['text']) > 220 else ''}\n")
            f.write(f"- {b}: {tb}{'...' if len(docs_map[b]['text']) > 220 else ''}\n\n")

    print(f"  wrote {out_path}  ({len(tp_sample)} TP + {len(fp_sample)} FP + {len(fn_sample)} FN)")

    # Also characterize FN by true-J distribution (answers "why missed?")
    fn_js = [jaccard(sh[a], sh[b]) for (a, b), _ in fn_pairs]
    fn_above_tau = sum(1 for j in fn_js if j >= 0.80)
    fn_below_tau = sum(1 for j in fn_js if j < 0.80)
    print(f"  FN audit: of {len(fn_pairs)} missed GT pairs, "
          f"{fn_below_tau} have true J < 0.80 (structural miss), "
          f"{fn_above_tau} have true J >= 0.80 (LSH filter miss)")


# ---------- pretty print + orchestration ----------

def _print_row(r: dict) -> None:
    print(
        f"  {r.get('label','?'):20s}  "
        f"cand={r['num_candidates']:7d}  "
        f"above={r['num_above_tau']:5d}  "
        f"R={r['overall_recall']*100:5.1f}%  "
        f"Pstrict={r['precision_strict_vs_gt']*100:5.1f}%  "
        f"final={r['final_corpus_size']:,}"
    )


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Lazy-load: first call triggers loads + prints.
    docs_by_id()
    print(f"[exp] survivors={len(docs_by_id())}  gt_pairs={len(gt_pairs())}")

    # Baseline — should match Phase 6 exactly.
    print("\n========== BASELINE  (word-3, k=64, (16,4), tau=0.80) ==========")
    baseline = run_config(3, "word", 64, 16, 4, 0.80, label="baseline")
    _print_row(baseline)

    exp_D = exp_D_threshold_sweep()
    exp_C = exp_C_banding_sweep()
    exp_B = exp_B_signature_length()
    exp_A = exp_A_shingle_type()

    manual_inspection()

    report = {
        "baseline": baseline,
        "exp_A_shingle_type": exp_A,
        "exp_B_signature_length": exp_B,
        "exp_C_banding": exp_C,
        "exp_D_threshold": exp_D,
    }
    out_path = OUTPUTS_DIR / "experiments_report.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[exp] full report -> {out_path}")


if __name__ == "__main__":
    main()

