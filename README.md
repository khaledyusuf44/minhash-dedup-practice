# minhash-dedup-practice

Implementing near-duplicate detection from scratch — shingling + MinHash + LSH — on a 100K-document corpus. No external dedup library; every piece is written to understand *why* it's there and what tradeoff it exposes.

This is a pretraining-data hygiene exercise. The direct follow-on is running this pipeline on the Common Crawl extracts in `../tiny_llm/crawl/` before any LLM training.

## Headline results

Pipeline: exact dedup → word-3-grams → 64-hash MinHash → (16, 4) LSH → exact-Jaccard verification at τ = 0.80 → union-find clusters → keep-longest.

| | |
|---|---:|
| Input corpus | 100,000 |
| Phase 2 exact-dedup removed | 4,124 |
| Phase 6 near-dedup removed | 5,844 |
| **Final deduplicated corpus** | **90,032** |
| Ground-truth recall (overall) | 60.4% |
| Precision against ground-truth | 75.0% |
| Full end-to-end wall time | ~15 seconds |

Four parameter sweeps (see `outputs/report.md` for full tables):

- **Shingle type is the biggest lever.** Switching word-3 → char-5 lifts recall from 60% to **94%**.
- **Signature length k is just a memory knob.** Final recall is unchanged across k ∈ {32, 64, 128} because verification uses exact Jaccard, not the MinHash estimate.
- **Banding (b, r) controls cost, not correctness.** (32, 2) generates 15× more candidates than (16, 4) and lands at identical recall.
- **Threshold τ is the main precision/recall knob.** τ=0.70 buys +20 pp recall for −4 pp precision vs τ=0.80.

And a sharp diagnostic: of 5,947 ground-truth pairs missed at the default configuration, **5,946 have true word-3 Jaccard below 0.80** — the threshold rejects them, not the LSH filter. LSH's filter error is ~0.02%.

## The 8 phases

Each has its own source file and its own documented contract:

| Phase | File | What it does |
|---|---|---|
| 1 | `src/build_corpus.py` | Build 100K AG News corpus + 15K injected duplicates across 6 edit types, with full ground truth |
| 2 | `src/preprocess.py` | `normalize()` + `exact_dedup()` using SHA-256 on normalized text |
| 3 | `src/shingling.py` | `shingle()` (word or char k-grams) + `jaccard()` |
| 4 | `src/minhash.py` | 31-bit universal hash family + signature matrix (N × 64, uint64) |
| 5 | `src/lsh.py` | Banding + bucket hashing + candidate-pair emission with shared-bands count |
| 6 | `src/dedup_pipeline.py` | Jaccard verification + union-find clustering + keep-rule + deduped corpus |
| 7 | `src/experiments.py` | Four sweeps (shingle, k, banding, τ) + manual-inspection sampler |
| 8 | `outputs/report.md` | One-page writeup (goal · corpus · method · results · error analysis · reflection) |

Full plan with math, mental models, and per-phase wiring to the big goal: **[PLAN.md](PLAN.md)**.

## Repository layout

```
├── PLAN.md                          # the full phase plan (the design doc)
├── README.md                          # this file
├── src/
│   ├── build_corpus.py                # phase 1
│   ├── preprocess.py                  # phase 2
│   ├── shingling.py                   # phase 3
│   ├── minhash.py                     # phase 4
│   ├── lsh.py                         # phase 5
│   ├── dedup_pipeline.py              # phase 6
│   └── experiments.py                 # phase 7
├── data/
│   ├── documents_100k.jsonl           # the corpus (100,000 rows)
│   ├── ground_truth_pairs_100k.csv    # 15,000 injected near-duplicate pairs
│   ├── documents_100k_after_exact.jsonl          # phase-2 survivors
│   └── documents_100k_after_exact_aggressive.jsonl
└── outputs/
    ├── report.md                      # THE writeup (start here)
    ├── manual_inspection.md           # 20 TP + 20 FP-like + 20 FN pairs
    ├── minhash_signatures.npy         # 95,876 × 64 uint64 signature matrix (49 MB)
    ├── minhash_coefficients.npz       # hash coefficients (reproducible)
    ├── candidate_pairs.csv            # phase 5 output — 22,532 pairs
    ├── duplicate_pairs.csv            # phase 6 output — verified pairs + J + decision
    ├── deduped_documents.jsonl        # phase 6 output — 90,032 docs (the deliverable)
    ├── experiments_report.json        # phase 7 sweeps
    └── *.json                         # per-phase metrics
```

## Reproduce

Everything is deterministic under `seed=0`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib tqdm datasets

# Phase 1 — build corpus + ground truth (downloads AG News on first run)
python src/build_corpus.py --target 100000

# Phases 2-6 — run the pipeline
python src/preprocess.py           # exact dedup, conservative + aggressive
python src/shingling.py            # Jaccard stats + corpus shingle budget
python src/minhash.py              # signature matrix + estimator validation
python src/lsh.py                  # candidate pairs at default (16, 4)
python src/dedup_pipeline.py       # verification + cluster + deduped output

# Phase 7 — four parameter sweeps + manual inspection (~75s)
python src/experiments.py

# Phase 8 — the writeup is already at outputs/report.md
```

## Start here

- **For the results**: [`outputs/report.md`](outputs/report.md) — one-page writeup.
- **For the design**: [`PLAN.md`](PLAN.md) — the full phase plan with math, mental models, and files-per-phase.
- **For the code**: start with [`src/minhash.py`](src/minhash.py) — the implementation where the universal hash family + overflow-safe numpy arithmetic lives.
- **For eyeballing duplicates**: [`outputs/manual_inspection.md`](outputs/manual_inspection.md) — 60 side-by-side pairs.

## What not to do (from the plan)

- Don't jump to 100K before pilot-testing at 100 / 1,000 / 10,000.
- Don't skip exact dedup — it's cheap and strictly subsumed by MinHash.
- Don't use a ready-made MinHash library for the core pipeline; the project *is* the implementation.
- Don't trust candidate pairs without exact-Jaccard verification; LSH is a filter, not a decision.
- Don't trust a single τ. Sweep it and inspect examples around the boundary.
- Don't merge duplicates pairwise without union-find; triangles need graph-level logic.
