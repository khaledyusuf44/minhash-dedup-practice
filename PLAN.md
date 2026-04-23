# Project plan

Design doc for the MinHash dedup pipeline — phase-by-phase, with math, mental models, and the file each phase ships.

## Project intent

Implement near-duplicate document detection from scratch using **shingling + MinHash + LSH banding**, then run it on a 100K-document corpus. The aim is *not* to build the fastest dedup — it is to understand, end to end, **why** each piece of the pipeline exists, what mathematical guarantee it provides, and what tradeoff knob it exposes.

This is a pretraining-data hygiene exercise. The direct follow-on is using this pipeline on the Common Crawl samples in `../tiny_llm/crawl/` before any model training.

## Big-goal wiring (why this sits next to `tiny_llm/`)

In real LLM pretraining, duplicate data silently damages the model: it inflates memorization, biases gradient updates toward repeated patterns, wastes compute, and contaminates eval. Exact hashing only catches byte-identical copies. Near-duplicate detection catches the cases that matter more in practice — the same article reposted with minor edits, template boilerplate, paraphrased copies, scraped-then-rewritten text.

At scale the pipeline can't afford two things: it can't store full documents for pairwise comparison, and it can't compare all pairs. MinHash solves the first (compact signatures that preserve Jaccard similarity); LSH solves the second (avoid comparing pairs that are obviously far apart). You will build both, from first principles, so that when you run dedup on Common Crawl later the knobs are not mysterious.

## Repository layout

```
data/       raw_documents.jsonl         (source corpus)
            documents_100k.jsonl        (final corpus with injected duplicates)
outputs/    minhash_signatures.npy      (N × k signature matrix)
            candidate_pairs.csv         (LSH-generated candidates)
            duplicate_pairs.csv         (candidates + exact Jaccard + decision)
            deduped_documents.jsonl     (final deduped corpus)
            report.md                   (writeup)
src/        build_corpus.py             (Phase 1)
            preprocess.py               (Phase 2)
            shingling.py                (Phase 3)
            minhash.py                  (Phase 4)
            lsh.py                      (Phase 5)
            dedup_pipeline.py           (Phase 6 — wires everything together)
notebooks/  minhash_dedup_practice.ipynb
```

All directories exist but are empty — the project is at step zero.

## Environment

- Create a fresh venv: `python3 -m venv .venv && source .venv/bin/activate`
- Base stack: `pip install numpy pandas matplotlib tqdm`
- Optional later: `pip install jupyter polars`
- **Do not** `pip install datasketch` or any other ready-made MinHash library for the core implementation. The point is to write the math yourself. External libraries are only acceptable later for cross-validation.

## How to run

Not wired up yet. Target invocation once `src/dedup_pipeline.py` exists:

```
python src/dedup_pipeline.py --input data/documents_100k.jsonl --out outputs/
```

Expected to produce `minhash_signatures.npy`, `candidate_pairs.csv`, `duplicate_pairs.csv`, `deduped_documents.jsonl`.

---

## The 8 phases

Each phase below has the same structure so the plan stays legible end-to-end:

- **Goal** — one sentence.
- **Subtasks** — the concrete pieces of work.
- **Mental model** — the frame to hold while coding.
- **Math / key ideas** — the formula or invariant that makes this phase work.
- **Files & core functions** — what you write, and what each piece contributes.
- **Big-goal wiring** — what this phase buys you in a real pretraining pipeline.
- **Deliverable** — the artifact that marks the phase done.

### Phase 1 — Corpus construction

**Goal.** Produce `data/documents_100k.jsonl`, ~100K short-to-medium documents, with a controlled mix of unique / exact-duplicate / near-duplicate rows so that later phases have ground truth.

**Subtasks.**
1. Pick a base corpus (a public dataset of short paragraphs/news snippets/Wikipedia leads works well; keep English-only for v1).
2. Inject controlled duplicates on top of the base:
   - exact duplicates (copy row verbatim)
   - punctuation/casing edits
   - typo edits
   - word substitutions (1–3 words)
   - sentence reordering within the doc
   - prefix/suffix additions (template framing)
3. Track injected pairs in a side file `data/ground_truth_pairs.csv` — columns `doc_id_1, doc_id_2, edit_type`. This is the recall denominator later.
4. Pilot first: 100 → 1,000 → 10,000 → 100,000. Do not jump to 100K blind.

**Mental model.** You are building a stress test with known answers. Every "too easy" near-dup is a wasted slot; every edit type should plausibly appear in Common Crawl.

**Math / key ideas.**
- Target duplicate rate around 5–15% (realistic for web corpora).
- Document length 1–20 sentences (short enough that shingling gives a useful signal; long enough that MinHash isn't dominated by noise).
- A document of `n` tokens yields `n - k + 1` distinct k-shingles — plan `n` so this count stays in the tens.

**Files & core functions.**
- `src/build_corpus.py`
  - `load_base_corpus(path) -> Iterator[str]` — source documents.
  - `inject_duplicates(docs, rates: dict) -> (docs, ground_truth_pairs)` — where `rates` carries the per-edit-type ratios.
  - `write_jsonl(path, records)` — writes `{doc_id, text}` rows.
- Output: `data/documents_100k.jsonl`, `data/ground_truth_pairs.csv`.

**Big-goal wiring.** Real pretraining corpora are already "polluted" in all these ways — templated pages, boilerplate, reposted news, minor edits between crawl snapshots. Your injected distribution is a controllable model of that pollution. Without ground truth, precision/recall in Phase 7 is guesswork.

**Deliverable.** `data/documents_100k.jsonl` (100K rows) + `data/ground_truth_pairs.csv`.

### Phase 2 — Preprocessing + exact-dedup baseline

**Goal.** Define a stable normalization, then remove byte-identical (after normalization) documents as a baseline — MinHash runs *after* this step, not instead of it.

**Subtasks.**
1. Implement `normalize(text) -> str`: lowercase, collapse whitespace, strip, optional punctuation strip (keep this as a flag — run it both ways in Phase 7).
2. Hash normalized text (SHA-256 or Python `hash` with a fixed seed); group identical hashes.
3. Print: exact-dup group count, largest group size, rows removed. Save the survivors before moving on.
4. Inspect 20 sample docs raw vs normalized. Ask: did I crush useful signal? Did I expose real duplicates that byte-level would have missed?

**Mental model.** Normalization is the only lever that can make two different documents look identical — be deliberate and log its effect. Exact dedup is always Phase 2 because it's trivially cheap and strictly subsumed by Jaccard-based dedup; skipping it means you later pay MinHash cost on trivially-identical rows.

**Math / key ideas.**
- Cryptographic hashes: collision probability negligible for N ≤ 10⁹ — exact dedup on hashes is effectively exact.
- Normalization composes: `dedup(normalize(x)) ⊇ dedup(x)`. The more aggressive the normalization, the more you collapse — and the more false merges you risk.

**Files & core functions.**
- `src/preprocess.py`
  - `normalize(text: str, strip_punct: bool = False) -> str`
  - `exact_dedup(records) -> (survivors, dup_groups)` — groups by `sha256(normalize(text))`.
- Output (intermediate): `outputs/exact_dedup_stats.json` or logged to stdout.

**Big-goal wiring.** On Common Crawl-style data, exact-dedup alone typically removes 20–40% of rows. Everything after this phase operates on a smaller, cleaner set — much cheaper for MinHash.

**Deliverable.** A normalized, exact-deduped survivor set in memory or an intermediate file; a stats printout.

### Phase 3 — Shingling + exact Jaccard baseline

**Goal.** Represent each document as a set of shingles, and implement exact Jaccard similarity so you have the ground-truth similarity that MinHash is approximating.

**Subtasks.**
1. Implement `shingle(text, k=3, mode="word") -> set[str]`. Start with word 3-grams.
2. Hand-validate on 5 documents — print the shingle set, check boundaries.
3. Implement `jaccard(A, B) -> float`.
4. Sanity pairs: exact copy → ~1.0; paraphrase → mid; unrelated → low. If the numbers look wrong, the shingler is wrong.

**Mental model.** Two shifts happen here: **document → set** (order-blind within k-grams) and **similarity → set similarity** (not semantic, not embedding — just token overlap). Everything MinHash does is an approximation of *this* Jaccard, not of any deeper meaning.

**Math / key ideas.**
- k-shingle set: `S_k(D) = { D[i:i+k] : 0 ≤ i ≤ |D| - k }`.
- Jaccard similarity: `J(A, B) = |A ∩ B| / |A ∪ B|`, with `J ∈ [0, 1]`, `J = 1 ⇔ A = B`, `J = 0 ⇔ A ∩ B = ∅`.
- Shingle choice encodes the notion of similarity:
  - small k (char 5-grams, word 2-grams) → robust to small edits but noisy on short text
  - larger k (word 5-grams) → stricter, misses paraphrases
- All-pairs Jaccard at N=100K is ~5×10⁹ pair comparisons — prohibitive. This is what forces MinHash + LSH.

**Files & core functions.**
- `src/shingling.py`
  - `shingle(text, k=3, mode="word") -> set[str]`
  - `jaccard(A: set, B: set) -> float`
- Output (optional): a `shingle_stats.json` with per-doc shingle counts so Phase 4 can budget hashing work.

**Big-goal wiring.** Jaccard on shingles is the de facto similarity metric for pretraining dedup (C4, RefinedWeb, The Pile all use it). Even giant pipelines fall back to this exact definition for the "verify candidate" step — they never abandon it, they only avoid computing it N² times.

**Deliverable.** `shingle` and `jaccard` functions, validated on hand-crafted pairs.

### Phase 4 — MinHash from scratch

**Goal.** Produce a `k`-dimensional integer signature per document such that the fraction of agreeing positions between two signatures approximates their Jaccard.

**Subtasks.**
1. Map shingles → integer IDs (a stable per-shingle hash is cheaper than a global dictionary at 100K scale).
2. Generate `k` hash functions from a universal family. Default `k = 64`; validate with `k = 128` later.
3. For each document D and each hash function `h_i`, compute `sig_i(D) = min{ h_i(s) : s ∈ S(D) }`.
4. Stack signatures into an `N × k` NumPy array; save as `.npy`.
5. Validate approximation on 20 known pairs: plot estimated vs true Jaccard — points should cluster around the diagonal.

**Mental model.** A MinHash is a **random coupon** from the shingle set. Two documents share the coupon exactly when the minimum under a random permutation lands inside their intersection — which happens with probability `|A ∩ B| / |A ∪ B|` = Jaccard. You repeat the experiment `k` times and average. That's the whole trick.

**Math / key ideas.**
- **Core theorem:** for a uniformly random permutation `π` over the shingle universe,
  `Pr[ min π(A) = min π(B) ] = J(A, B)`.
- **Estimator:** given `k` independent hash functions, `Ĵ(A, B) = (1/k) · Σ_i [ sig_i(A) = sig_i(B) ]`.
- **Unbiasedness:** `E[Ĵ] = J`. **Variance:** `Var[Ĵ] = J(1 − J) / k`. At `J = 0.5, k = 64`, standard error ≈ `√(0.25/64) ≈ 0.0625` (≈ 6 pp). At `k = 128`, ≈ 4.4 pp.
- **Universal hash family:** `h_i(x) = (a_i · x + b_i) mod p`, with `p` prime and `> max_shingle_id`; `a_i ∈ [1, p−1]`, `b_i ∈ [0, p−1]` drawn at init with a fixed seed.
- **Why `min`:** the min is permutation-invariant over ties and lets you estimate Jaccard without materializing intersections.

**Files & core functions.**
- `src/minhash.py`
  - `make_hashes(k: int, seed: int = 0) -> (a, b, p)` — draws hash coefficients.
  - `signature(shingle_ids: np.ndarray, a, b, p) -> np.ndarray` — length-`k` signature; the hot loop.
  - `signatures_for_corpus(docs, k=64) -> np.ndarray` — N×k matrix.
  - `estimate_jaccard(sig_A, sig_B) -> float` — agreement ratio.
- Output: `outputs/minhash_signatures.npy`, `outputs/doc_id_to_row.csv`.

**Big-goal wiring.** This is the one phase where you're replacing expensive exact math with a cheap probabilistic approximation. In pretraining-scale dedup, this replacement is not optional — storing full shingle sets for 10⁹ documents is infeasible, storing 64 ints per document is ~256 bytes × 10⁹ ≈ 256 GB, manageable.

**Deliverable.** N × 64 signature matrix; a small validation plot (estimated vs true J on ~100 pairs) that shows the cloud sits near the diagonal.

### Phase 5 — LSH banding + candidate generation

**Goal.** Find candidate pairs without comparing all pairs. Split signatures into bands, hash bands into buckets, and emit pairs that collide in at least one bucket.

**Subtasks.**
1. Choose a `(b, r)` split with `b · r = k`. Default `b = 16, r = 4` at `k = 64`.
2. For each document and each of the `b` bands, hash the `r`-tuple (slice of the signature) into a bucket. A simple `hash(tuple(slice))` works.
3. Group doc IDs by `(band_idx, bucket)`. Within each group, emit all pairs.
4. Deduplicate pairs globally (the same pair may collide in multiple bands — that's a useful signal, store the count).
5. Log: total candidates generated, reduction ratio vs all-pairs.

**Mental model.** Banding is a tunable OR of ANDs. "All `r` positions within a band must match" = AND (rare for low-similarity pairs). "Any of the `b` bands must match" = OR (amplifies the chance for high-similarity pairs). Choose `b, r` to sharpen this S-curve around the similarity threshold you care about.

**Math / key ideas.**
- **Collision probability for a single band:** `P(band match | J = s) = s^r`.
- **Probability of colliding in at least one band:** `P(candidate | J = s) = 1 − (1 − s^r)^b`.
- This is an **S-curve** in `s`. The inflection (informal threshold) sits near `s* ≈ (1/b)^(1/r)`.
- Planning table at `k = 64`:
  - `b=16, r=4` → `s* ≈ 0.500` (permissive; many candidates)
  - `b=8, r=8` → `s* ≈ 0.760` (strict; fewer candidates, risks missing moderate dups)
  - `b=32, r=2` → `s* ≈ 0.177` (very permissive; candidate explosion)
- Tuning rule: set `s*` slightly **below** your intended final Jaccard threshold `τ` from Phase 6. You want recall at the candidate stage; precision comes from exact Jaccard verification.

**Files & core functions.**
- `src/lsh.py`
  - `band_signatures(sig_matrix: np.ndarray, b: int, r: int) -> np.ndarray` — reshape to N×b×r.
  - `bucket_band(band_slices) -> dict[bucket_key, list[doc_idx]]`.
  - `candidate_pairs(signatures, b, r) -> Iterator[(doc_id_1, doc_id_2, band_count)]`.
- Output: `outputs/candidate_pairs.csv` with columns `doc_id_1, doc_id_2, shared_bands`.

**Big-goal wiring.** The reduction ratio is the number that gets you from O(N²) to something runnable on a laptop. On 100K documents you want to see candidate count in the low millions at most — not 5 × 10⁹. If your banding setup gives you 100K candidates, you missed too much; 100M and you tuned too loose.

**Deliverable.** `outputs/candidate_pairs.csv`; a printed reduction ratio (candidates / N-choose-2).

### Phase 6 — Verification, thresholding, dedup decisions

**Goal.** Turn candidate pairs into an actual deduplicated corpus. Verify each candidate with exact Jaccard, apply a threshold, pick a keep-rule, emit the survivor file.

**Subtasks.**
1. For each candidate, recompute exact Jaccard on the full shingle sets. This is why shingles per document must be reachable (keep them in memory, or recompute from text on demand).
2. Apply threshold `τ` (default `0.80`). Mark pairs above `τ` as duplicate.
3. Build a union-find over duplicate pairs; each connected component is a duplicate cluster.
4. Apply a keep-rule per cluster: first-seen / longest / cleanest. **Pick one and state it**; do not mix.
5. Emit `outputs/deduped_documents.jsonl` and `outputs/duplicate_pairs.csv`.

**Mental model.** Duplicates are a graph, not a list. Three-way near-duplicates form a triangle in the similarity graph and collapse into one surviving document — not into three pair decisions.

**Math / key ideas.**
- Final decision metric is **exact Jaccard**, not MinHash estimate. MinHash + LSH is a candidate filter; the filter can be noisy. The verifier must not be.
- `τ` sets the operating point on the S-curve. Typical values:
  - `τ = 0.9` — conservative; only near-copies.
  - `τ = 0.8` — standard for pretraining (Gopher, RefinedWeb region).
  - `τ = 0.7` — aggressive; starts catching paraphrases but also template collisions.
- Keep-rule introduces a **bias**: "keep longest" biases the training distribution toward long-form; "keep first-seen" biases toward whatever ordering the input file had. The bias is unavoidable — just make it explicit.
- Union-find: nearly-linear amortized complexity (`α(N)`); trivial to implement.

**Files & core functions.**
- `src/dedup_pipeline.py`
  - `verify_candidates(candidates, shingles_by_doc, tau) -> duplicate_pairs`
  - `build_clusters(duplicate_pairs) -> list[set[doc_id]]` — union-find.
  - `select_survivors(clusters, docs, rule="longest") -> set[doc_id]`
  - `write_deduped(survivors, docs, path)`
- Output: `outputs/duplicate_pairs.csv` (`doc_id_1, doc_id_2, jaccard, decision`), `outputs/deduped_documents.jsonl`, and a counts summary (original / exact removed / near removed / final).

**Big-goal wiring.** What leaves this phase is *the actual training set*. Every downstream training run inherits the biases of your keep-rule and threshold. Writing them down is how you avoid mysterious eval distribution shifts three weeks later.

**Deliverable.** `outputs/deduped_documents.jsonl` + counts + the keep-rule documented in `report.md`.

### Phase 7 — Evaluation, inspection, experiments

**Goal.** Characterize the dedup's behavior — not just its headline numbers. Measure precision/recall against injected ground truth; manually inspect false positives and false negatives; run the three tuning experiments.

**Subtasks.**
1. **Precision/recall** vs `data/ground_truth_pairs.csv`.
2. **Manual inspection**, at least 30–50 pairs total across: true positives above `τ`, false positives (above `τ` but intuitively not duplicates), false negatives (injected pairs missed entirely).
3. **Experiment A — shingle type:** word 3-grams vs word 5-grams vs char 5-grams. Report P/R for each.
4. **Experiment B — signature length:** `k = 32, 64, 128`. Report estimator error vs speed/memory.
5. **Experiment C — banding:** `(16,4)` vs `(8,8)` vs `(32,2)`. Report candidates generated and final P/R.
6. **Experiment D — threshold sweep:** `τ ∈ {0.7, 0.8, 0.9}`. Report the P/R tradeoff curve.

**Mental model.** A single accuracy number always hides the shape of the error. Short templated text drives most false positives; aggressive word-substitution edits drive most false negatives. The point of the experiments is to attribute each class of failure to a tunable knob.

**Math / key ideas.**
- `Precision = |detected ∩ true| / |detected|`, `Recall = |detected ∩ true| / |true|`, `F1 = 2PR / (P+R)`.
- **Expected failure modes:**
  - False positives: short texts (few shingles → small-denominator Jaccard spikes easily), template prefixes/suffixes, aggressive normalization.
  - False negatives: heavy reordering, many word substitutions, k chosen too large for the document length, `(b, r)` set too strict.
- Each experiment changes *one* knob so the P/R delta is attributable.

**Files & core functions.**
- In `notebooks/minhash_dedup_practice.ipynb`: the experiment grid and plots.
- Optional: `src/eval.py` with `precision_recall(detected_pairs, ground_truth_pairs)`.

**Big-goal wiring.** The experiments map directly to the knobs you will tune on Common Crawl in `../tiny_llm/`. The intuition that 0.8 is usually right, that `(16, 4)` is usually right, that short text is the dominant FP source — that's what survives this project and saves you time later.

**Deliverable.** An experiment table in the report + example pairs for each failure class.

### Phase 8 — Writeup

**Goal.** A one-page `outputs/report.md` with six sections: **Goal · Corpus · Method · Results · Error analysis · Reflection**.

**Subtasks.**
1. State the final setup (shingle choice, `k`, `(b, r)`, `τ`, keep-rule) explicitly — a future reader should be able to reproduce it.
2. Numbers: original count, exact-dup removed, near-dup removed, final count, P/R if measured.
3. Three concrete failure examples with evidence.
4. Three concrete tradeoffs named with the knob that controls them.
5. One paragraph on what you'd change when running this on Common Crawl.

**Mental model.** Same rule as `../language-id-practice/REPORT.md`: headline number → per-axis breakdown → error analysis → reflection. Do not lead with the headline, and do not skip the reflection.

**Deliverable.** `outputs/report.md`.

---

## Working rhythm

The plan groups the phases into 5 sessions, same rhythm as `../language-id-practice/`:

- **Session 1:** Phase 1 + Phase 2 (corpus + preprocessing + exact dedup).
- **Session 2:** Phase 3 + small-sample Phase 4 (shingling, Jaccard, first signatures on ~1K docs).
- **Session 3:** Phase 4 at full scale + Phase 5 (signatures on 100K, banding).
- **Session 4:** Phase 6 (verification + dedup decisions).
- **Session 5:** Phase 7 + Phase 8 (experiments + report).

Finish a session before starting the next. Do not half-finish multiple.

## Definition of done

- `outputs/deduped_documents.jsonl` exists and is smaller than the input.
- `outputs/duplicate_pairs.csv` contains `jaccard` and `decision` columns, populated.
- Precision and recall numbers reported against injected ground truth.
- At least 30–50 duplicate pairs manually inspected, with examples of TP / FP / FN in the report.
- Three experiments from Phase 7 actually run (not just planned).
- Report names at least three concrete failure modes with evidence.

## Anti-patterns

- Don't jump to 100K before the 100 / 1,000 / 10,000 pilots work.
- Don't skip exact dedup — it's cheap and strictly subsumed by MinHash, which means skipping it just makes MinHash waste work.
- Don't use `datasketch` or any other MinHash library for the core pipeline. The project is the implementation.
- Don't trust candidate pairs without exact Jaccard verification — LSH is a filter, not a decision.
- Don't trust a single `τ`. Sweep it and inspect a few pairs around the boundary.
- Don't merge duplicates pairwise without union-find — you will produce inconsistent decisions on triangles.
- Don't lose track of which edits exist in ground truth vs which you're detecting. Without that, P/R is storytelling.
