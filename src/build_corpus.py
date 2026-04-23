"""Phase 1: build a corpus with controlled injected near-duplicates + ground truth.

Pulls base documents from AG News (news snippets, 1-3 sentences, English, ~120K rows).
Injects six edit types on top and records every (original, duplicate) pair so later
phases can measure precision/recall.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path


def load_base_corpus(n_needed: int) -> list[str]:
    from datasets import load_dataset  # heavy import — keep local

    ds = load_dataset("ag_news", split="train")
    if n_needed > len(ds):
        raise ValueError(f"Need {n_needed} unique rows; AG News train only has {len(ds)}")
    return [ds[i]["text"] for i in range(n_needed)]


# ---------- edit operations ----------
# Each takes (text, rng) and returns an edited string. Parameters chosen so that
# the expected word-3-gram Jaccard with the original lands in a useful band:
#   exact         ~ 1.00 (identical; preprocessing-dependent if we change casing)
#   punctuation   ~ 0.0  raw / ~1.0 after normalization  (tests preprocessing)
#   typo          ~ 0.75-0.85  (a couple of invalidated shingles)
#   word_sub      ~ 0.60-0.75  (3 subs invalidate ~9 shingles)
#   reorder       depends on sentence count; within-sentence shingles preserved
#   prefix_suffix ~ 0.85-0.95  (adds a few shingles, near-identical body)


SYNONYMS = {
    "said": "stated", "stated": "said",
    "big": "large", "large": "big",
    "small": "little", "little": "small",
    "fast": "quick", "quick": "fast",
    "house": "home", "home": "house",
    "car": "vehicle", "vehicle": "car",
    "buy": "purchase", "purchase": "buy",
    "good": "great", "great": "good",
    "bad": "poor", "poor": "bad",
    "happy": "glad", "glad": "happy",
    "start": "begin", "begin": "start",
    "end": "finish", "finish": "end",
    "use": "utilize", "utilize": "use",
    "show": "display", "display": "show",
    "help": "assist", "assist": "help",
    "find": "locate", "locate": "find",
    "make": "create", "create": "make",
    "get": "obtain", "obtain": "get",
    "important": "crucial", "crucial": "important",
    "right": "correct", "correct": "right",
    "talk": "speak", "speak": "talk",
    "company": "firm", "firm": "company",
    "country": "nation", "nation": "country",
    "begin": "start", "end": "close",
    "people": "individuals", "individuals": "people",
    "said": "stated",
    "quick": "rapid", "rapid": "quick",
    "large": "sizeable",
    "buy": "acquire", "acquire": "buy",
    "price": "cost", "cost": "price",
    "report": "account", "account": "report",
    "win": "triumph", "triumph": "win",
    "lose": "forfeit",
    "plan": "scheme", "scheme": "plan",
    "issue": "matter", "matter": "issue",
    "group": "team", "team": "group",
}

PREFIXES = [
    "Breaking news: ",
    "Update: ",
    "Report: ",
    "Exclusive: ",
    "In today's news, ",
    "According to sources, ",
    "It has been reported that ",
]

SUFFIXES = [
    " Read more.",
    " (via Reuters)",
    " -- developing story.",
    " Click here for updates.",
    " Stay tuned.",
    " More to follow.",
]

_TRAIL_PUNCT = ".,!?;:\"'"


def edit_exact(text: str, rng: random.Random) -> str:
    return text


def edit_punctuation(text: str, rng: random.Random) -> str:
    out = text.lower() if rng.random() < 0.5 else text.upper()
    if rng.random() < 0.5:
        out = out.replace(",", "")
    if rng.random() < 0.5:
        out = out.replace(".", "..")
    return out


def edit_typos(text: str, rng: random.Random, n_typos: int = 2) -> str:
    """Swap adjacent chars inside n_typos distinct long words. Retries if a
    chosen swap is a no-op (identical neighboring chars); falls back to a word
    swap if the document has no long words at all or if two applied swaps
    happen to cancel out."""
    words = text.split()
    if not words:
        return text
    original = words[:]
    long_idx = [i for i, w in enumerate(words) if len(w) >= 3]
    if not long_idx:
        return _swap_two_words(text, rng)
    typos_done = 0
    attempts = 0
    while typos_done < n_typos and attempts < 20:
        attempts += 1
        i = rng.choice(long_idx)
        w = words[i]
        j = rng.randrange(len(w) - 1)
        if w[j] == w[j + 1]:
            continue  # swap would be a no-op
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        typos_done += 1
    if typos_done == 0 or words == original:
        return _swap_two_words(text, rng)
    return " ".join(words)


def _swap_two_words(text: str, rng: random.Random) -> str:
    """Word-level fallback: swap two distinct word positions. Invalidates a few
    word-3-gram shingles without changing the bag of words, so the edit stays
    in the near-dup band (high Jaccard, not exact)."""
    words = text.split()
    if len(words) < 2:
        return text
    for _ in range(10):
        i, j = rng.sample(range(len(words)), 2)
        if words[i] != words[j]:
            words[i], words[j] = words[j], words[i]
            return " ".join(words)
    return text


def edit_word_substitution(text: str, rng: random.Random, n_subs: int = 3) -> str:
    words = text.split()
    indices = list(range(len(words)))
    rng.shuffle(indices)
    subs_done = 0
    for i in indices:
        if subs_done >= n_subs:
            break
        w = words[i]
        lw = w.lower().strip(_TRAIL_PUNCT)
        if lw in SYNONYMS:
            replacement = SYNONYMS[lw]
            if w[:1].isupper():
                replacement = replacement.capitalize()
            trail = ""
            for ch in reversed(w):
                if ch in _TRAIL_PUNCT:
                    trail = ch + trail
                else:
                    break
            words[i] = replacement + trail
            subs_done += 1
    if subs_done == 0:
        # No synonym match in this document — fall back to a word-pair swap.
        # Same ground-truth label (word-level distortion), same near-dup band.
        return _swap_two_words(text, rng)
    return " ".join(words)


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def edit_sentence_reorder(text: str, rng: random.Random) -> str:
    sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    if len(sents) >= 2:
        # With only 2 sentences, rng.shuffle has a 50% chance of producing the
        # same order — loop until we actually permute, then reverse as a last
        # resort if every shuffle came back identical.
        original = sents[:]
        for _ in range(10):
            rng.shuffle(sents)
            if sents != original:
                return " ".join(sents)
        sents.reverse()
        return " ".join(sents)
    # Single-sentence source — reverse a middle window of words so the edit is
    # still a reorder (just word-level, not sentence-level).
    words = text.split()
    if len(words) >= 4:
        n = len(words)
        start = rng.randrange(max(1, n // 3))
        end = min(n, start + max(3, n // 2))
        words[start:end] = words[start:end][::-1]
        return " ".join(words)
    return _swap_two_words(text, rng)


def edit_prefix_suffix(text: str, rng: random.Random) -> str:
    prefix = rng.choice(PREFIXES) if rng.random() < 0.6 else ""
    suffix = rng.choice(SUFFIXES) if rng.random() < 0.6 else ""
    if not prefix and not suffix:
        suffix = rng.choice(SUFFIXES)
    return prefix + text + suffix


EDIT_OPS = {
    "exact": edit_exact,
    "punctuation": edit_punctuation,
    "typo": edit_typos,
    "word_sub": edit_word_substitution,
    "reorder": edit_sentence_reorder,
    "prefix_suffix": edit_prefix_suffix,
}


# ---------- corpus assembly ----------

def inject_duplicates(
    uniques: list[str],
    counts: dict[str, int],
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    docs = [{"doc_id": str(i), "text": t} for i, t in enumerate(uniques)]
    ground_truth: list[dict] = []
    next_id = len(uniques)
    for edit_type, n in counts.items():
        op = EDIT_OPS[edit_type]
        # sample with replacement so the same source can appear multiple times
        # (realistic: a popular article gets reposted many times)
        sources = [rng.randrange(len(uniques)) for _ in range(n)]
        for src_idx in sources:
            new_text = op(uniques[src_idx], rng)
            new_id = str(next_id)
            docs.append({"doc_id": new_id, "text": new_text})
            ground_truth.append(
                {
                    "doc_id_1": str(src_idx),
                    "doc_id_2": new_id,
                    "edit_type": edit_type,
                }
            )
            next_id += 1
    return docs, ground_truth


def size_suffix(n: int) -> str:
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)


def build(target_size: int, seed: int, out_dir: Path, dup_rate: float = 0.15) -> None:
    rng = random.Random(seed)

    n_dups = int(round(target_size * dup_rate))
    n_unique = target_size - n_dups

    # Allocate the dup budget across the six edit types. The distribution below
    # gives exact/punct/typo/word_sub a bigger share (they are the common real-world
    # cases) and reorder/prefix_suffix a smaller share.
    shares = {
        "exact": 0.20,
        "punctuation": 0.20,
        "typo": 0.20,
        "word_sub": 0.20,
        "reorder": 0.10,
        "prefix_suffix": 0.10,
    }
    counts = {k: int(n_dups * v) for k, v in shares.items()}
    # Absorb rounding drift into exact duplicates.
    counts["exact"] += n_dups - sum(counts.values())

    print(f"[build] target={target_size}  unique={n_unique}  dups={sum(counts.values())}  dup_rate={dup_rate:.0%}")
    print(f"[build] edit mix: {counts}")

    uniques = load_base_corpus(n_unique)
    docs, ground_truth = inject_duplicates(uniques, counts, rng)

    # Shuffle so duplicates are not adjacent — realistic for web crawl ordering.
    rng.shuffle(docs)

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = size_suffix(target_size)
    docs_path = out_dir / f"documents_{suffix}.jsonl"
    gt_path = out_dir / f"ground_truth_pairs_{suffix}.csv"

    write_jsonl(docs_path, docs)
    write_csv(gt_path, ground_truth, fieldnames=["doc_id_1", "doc_id_2", "edit_type"])

    # Summary report.
    lengths_words = [len(d["text"].split()) for d in docs]
    lengths_chars = [len(d["text"]) for d in docs]
    print(f"[build] wrote {len(docs)} docs  -> {docs_path}")
    print(f"[build] wrote {len(ground_truth)} ground-truth pairs  -> {gt_path}")
    print(
        f"[build] doc-length stats:  words  min={min(lengths_words)}  "
        f"median={sorted(lengths_words)[len(lengths_words)//2]}  "
        f"max={max(lengths_words)}  |  chars median={sorted(lengths_chars)[len(lengths_chars)//2]}"
    )
    print(f"[build] ground-truth by edit_type: {dict(Counter(r['edit_type'] for r in ground_truth))}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=int, default=100, help="total document count")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dup-rate", type=float, default=0.15)
    p.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "data"),
        help="output directory",
    )
    args = p.parse_args()
    build(args.target, args.seed, Path(args.out), args.dup_rate)


if __name__ == "__main__":
    main()
