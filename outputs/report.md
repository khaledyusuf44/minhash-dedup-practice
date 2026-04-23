# MinHash Deduplication — Practice Report

## 1. Goal

Implement shingling + MinHash + LSH near-duplicate detection **from scratch** on a 100K-document corpus, verify candidates with exact Jaccard, cluster with union-find, and characterize what the pipeline catches and misses under different parameter choices.

## 2. Corpus

**100,000 documents** assembled on top of AG News (news snippets, English, median 37 words / 233 chars per doc).

- 85,000 uniques drawn directly from AG News.
- **15,000 injected duplicates**, with full ground-truth tracking at `data/ground_truth_pairs_100k.csv`:
  - 3,000 exact (byte-identical copies)
  - 3,000 punctuation edits (case flips, comma/period mutation)
  - 3,000 typos (adjacent-character swaps, 2 per doc)
  - 3,000 word substitutions (3 per doc from a small synonym table, fallback word-swap)
  - 1,500 sentence reorders (with single-sentence fallback to word-window reversal)
  - 1,500 prefix/suffix additions ("Breaking news:", " (via Reuters)", etc.)

15% duplicate rate matches realistic web-crawl pollution. The surrounding AG News corpus also contained **~170 naturally-occurring byte-duplicates** before injection — useful as a second signal in Phase 7.

## 3. Method

Six-phase pipeline, implemented with no external MinHash library (`datasets` only used for corpus download):

| Phase | Operation | Key parameter |
|---|---|---|
| 2 | Exact dedup | `normalize = lowercase + collapse-whitespace`, SHA-256 grouping |
| 3 | Shingling | word **3-grams** on normalized text; exact Jaccard as ground-truth similarity |
| 4 | MinHash | **k = 64** hash functions, 31-bit universal family `h(x) = (ax + b) mod (2³¹ − 1)` |
| 5 | LSH banding | **b = 16, r = 4** → informal threshold s* ≈ 0.50 |
| 6 | Verify + cluster + decide | exact Jaccard threshold **τ = 0.80**; union-find; keep-rule = **longest text survives** |

Reasoning on a few choices:
- 31-bit Mersenne prime for `p` keeps the `a · id + b` product under 2⁶² so numpy uint64 arithmetic has no overflow surprises.
- Conservative normalization (no punctuation strip) for Phase 2 keeps aggressive merges away from exact-dedup — MinHash handles those.
- τ = 0.80 is the standard Gopher / RefinedWeb operating point. Phase 7 sweeps it to measure sensitivity.

## 4. Results

### Headline counts

| | |
|---|---:|
| Original corpus | 100,000 |
| Phase 2 (exact) removed | 4,124 |
| Phase 6 (near-dup) removed | 5,844 |
| **Final deduplicated corpus** | **90,032** |
| Total reduction | 9.97% |

### Ground-truth recall by edit type (default: word-3, k=64, (16,4), τ=0.80)

| Edit type | Caught | Missed | Recall | Notes |
|---|---|---|---|---|
| exact | 3000 | 0 | **100.0%** | all caught at Phase 2 |
| prefix_suffix | 1472 | 28 | **98.1%** | body dominates shingle set, J median 0.92 |
| punctuation | 2053 | 947 | 68.4% | split between Phase 2 and Phase 6 catches |
| word_sub | 1273 | 1727 | 42.4% | J median 0.76 — straddles τ=0.80 |
| reorder | 650 | 850 | 43.3% | J median 0.29 — structurally sub-threshold |
| typo | 605 | 2395 | 20.2% | J median 0.73 — mostly sub-threshold |
| **overall** | **9053** | **5947** | **60.4%** | |

Strict precision against GT: **75.0%**. The 25% "extra" detections include real natural AG-News duplicates (see §5).

### Estimator validation (Phase 4)

MinHash is unbiased and its empirical standard error matches theory:

| edit type | mean true J | mean Ĵ | bias (err_mean) | empirical err_std | theoretical √(J(1-J)/k) |
|---|---|---|---|---|---|
| prefix_suffix | 0.909 | 0.907 | −0.003 | 0.037 | 0.035 |
| punctuation | 0.776 | 0.775 | −0.001 | 0.048 | 0.048 |
| reorder | 0.523 | 0.522 | −0.001 | 0.049 | 0.049 |
| word_sub | 0.758 | 0.757 | −0.001 | 0.053 | 0.052 |
| typo | 0.723 | 0.724 | +0.001 | 0.054 | 0.055 |

### Experiment A — shingle type (k=64, (16,4), τ=0.80)

| shingle | candidates | above τ | **recall** | P_strict |
|---|---|---|---|---|
| word-3 | 22,532 | 6,000 | 60.4% | 75.0% |
| word-5 | 17,353 | 3,275 | 47.9% | 82.3% |
| **char-5** | 56,380 | 14,413 | **94.0%** | 67.6% |

**The biggest lever of the whole project.** Switching from word-3 to char-5 lifts recall 34 pp — character n-grams absorb word-level edits (typos, substitutions, reorderings) that word-3-grams cannot.

### Experiment B — signature length (k ∈ {32, 64, 128})

| k | candidates | final recall | err_mean | err_std | memory |
|---|---|---|---|---|---|
| 32 | 18,722 | 60.3% | −0.001 | 0.072 | 24.5 MB |
| 64 | 22,532 | 60.4% | −0.004 | 0.048 | 49.1 MB |
| 128 | 26,702 | 60.4% | 0.000 | 0.036 | 98.2 MB |

Empirical err_std tracks `√(0.25/k)` almost exactly. **Final recall is unchanged across k** because verification uses exact Jaccard, not Ĵ — k only affects what's filtered into the candidate bucket.

### Experiment C — LSH banding at k=64

| (b, r) | s* ≈ | candidates | above τ | recall |
|---|---|---|---|---|
| (16, 4) | 0.500 | 22,532 | 6,000 | 60.4% |
| (8, 8) | 0.760 | 9,602 | 5,543 | 58.3% |
| (32, 2) | 0.177 | 332,149 | 6,001 | 60.4% |

(32, 2) generates **15× more candidates** than (16, 4) but lands at identical recall — the extra 310K candidates were all below τ and rejected by the verifier. A classic "candidate explosion" at a low s*.

### Experiment D — threshold sweep (word-3, k=64, (16,4))

| τ | above τ | recall | P_strict |
|---|---|---|---|
| 0.70 | 10,776 | **80.8%** | 71.1% |
| 0.80 | 6,000 | 60.4% | 75.0% |
| 0.90 | 2,083 | 41.8% | **88.0%** |

The only knob that meaningfully trades recall for precision at fixed shingle/banding.

## 5. Error analysis

### Example true positives (all 4 caught correctly)

- **prefix_suffix (J=0.896)** — `5685 / 99815`: "Olympics: Germans Unhorsed..." vs "Breaking news: Olympics: Germans Unhorsed..."
- **typo (J=0.822)** — `11676 / 91644`: "...two of European football..." vs "...two of Euroepan ofotball..." (typos in "European" and "football")
- **word_sub (J=0.906)** — `84519 / 94318`: body identical, a few words swapped via the synonym table
- **punctuation (J=0.957)** — `58130 / 90737`: "Britons Must..." vs "britons must..." (casing only)

### Example "false positives" (above τ, not in GT — but most are natural duplicates)

- **`10595 / 10650`  J=0.857** — "The ranks of Americans fil**l**ing for..." vs "The ranks of Americans fil**ing** for..." (two real AG News filings of the same Reuters story, one had a typo in the source)
- **`32396 / 32431`  J=0.891** — "Iran Rejects UN Call for Uranium Enrichment Freeze" filed twice with minor copy edits
- **`30367 / 30430`  J=0.878** — "Hurricane Ivan Slams Gulf Coast; **22** Dead" vs "...; **23** Dead" — same story, different editions after the death toll was updated

These are correctly flagged as near-duplicates by the pipeline; they just aren't in our injection ground truth. **Strict precision is therefore an underestimate of true precision.**

### Example false negatives

- **word_sub (J=0.714)** — `67649 / 96305`: 6 words swapped across the first sentence ("Darfur peace talks open" → "Darfur peace talks Union's") pushes J below τ=0.80.
- **typo (J=0.613)** — `10641 / 92968`: 3 typos in a short headline ("Abu Ghraib" → "Abu Grhaib", "Wednesday" → "Wednesday"/"reudced") → many shingles broken in a doc with few to start with.
- **reorder (J=0.250)** — `67324 / 97900`: first sentence fully word-reversed ("LONDON, Oct 21... has decided not to discipline" → "skipper England discipline to not decided has Association..."). Word-3 shingles are destroyed; char-5 would have caught it.

### Which setup choice affected quality most

1. **Shingle type** — word-3 → char-5 moved recall from 60.4% to 94.0%. No other single change came close.
2. **Threshold τ** — inside a fixed shingle choice, τ controls everything. 0.70 → 0.90 swung recall from 81% to 42%.
3. **Banding (b, r)** — doesn't change final recall provided τ is applied to exact Jaccard; only controls candidate-count cost.

### One sharp diagnostic

Of the **5,947 GT pairs missed** at the default configuration, **5,946 have true word-3 Jaccard below 0.80** — the threshold is rejecting them, not LSH. Only **1** pair had true J ≥ 0.80 and was lost at the LSH filter stage (~0.02% filter error rate). LSH is essentially perfect as implemented; the recall story is entirely a threshold story.

## 6. Reflection

Three things surprised me in a useful way:

- **LSH's S-curve behaves like the formula says, to within sample noise.** At (16, 4) the predicted P(candidate) at J=0.7 was 0.988; we saw 97.4% typo recall at the LSH stage (mean true J 0.72). At J=0.5 the prediction was 0.644; reorder recall at the LSH stage (mean J 0.52) was 47%, below prediction because reorder has a wide bimodal distribution with p25 = 0.27.

- **Threshold, not LSH, is where near-dups get lost.** The instinct to tune (b, r) aggressively is misplaced once you already verify with exact J — (32, 2) just burns compute. The lever that actually matters is τ, and it's a direct P/R tradeoff.

- **Char-5 is the right default for a Common Crawl pipeline.** Word-shingles are a cleaner "same article" notion on short AG-News snippets, but char-5 is what catches the edits that matter in practice (typos, reformats, translation-style paraphrases). The precision hit from char-5 is mostly real near-dups we under-labeled, not false positives in any harmful sense.

What I'd change when running this on Common Crawl (the follow-on in `../tiny_llm/crawl/`):

1. Keep conservative normalization for Phase 2 — web crawl has many casing/whitespace-only variants that deserve a cheap prepass.
2. Switch shingling to char-5 or word-5 depending on whether I'm deduplicating at the "page" or "story" level. Char-5 for page-level, word-5 for story-level.
3. Target τ = 0.80 as the main threshold and always report two alternate-τ numbers alongside — 0.70 as an "aggressive" variant, 0.90 as a "conservative" one.
4. Always inject a small ground-truth set before running — otherwise recall is unmeasurable and you can't defend parameter choices.

## 7. Artifacts

- `data/documents_100k.jsonl`, `data/ground_truth_pairs_100k.csv` — corpus + injected GT
- `data/documents_100k_after_exact.jsonl` — Phase 2 survivors (95,876 docs)
- `outputs/minhash_signatures.npy`, `outputs/minhash_coefficients.npz` — Phase 4 signature matrix (49 MB) and reproducible hash coefficients
- `outputs/candidate_pairs.csv` — Phase 5 LSH output (22,532 pairs)
- `outputs/duplicate_pairs.csv`, `outputs/deduped_documents.jsonl` — Phase 6 outputs
- `outputs/exact_dedup_stats_*.json`, `outputs/shingle_stats_*.json`, `outputs/lsh_stats*.json`, `outputs/dedup_summary.json` — per-phase metrics
- `outputs/experiments_report.json` — full sweep results for Experiments A–D
- `outputs/manual_inspection.md` — 20 TP + 20 "FP" + 20 FN pairs side-by-side
- `src/{build_corpus, preprocess, shingling, minhash, lsh, dedup_pipeline, experiments}.py` — one file per phase
