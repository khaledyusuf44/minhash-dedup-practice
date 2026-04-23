"""Phase 2: text normalization + exact-duplicate removal (the pre-MinHash baseline).

Two normalization levels:

  conservative : lowercase + whitespace collapse (what Gopher/RefinedWeb/The Pile
                 typically do before MinHash — minimal risk of false merges)
  aggressive   : the above + strip punctuation (catches more, but risks merging
                 documents whose punctuation actually carries signal)

This script runs both on the input corpus and cross-checks each catch against
the Phase 1 ground-truth file so we can see which edit types get absorbed at
each normalization level.

Main downstream artifact: data/<stem>_after_exact.jsonl (conservative survivors)
— this is what Phase 3 shingles. The aggressive file is for comparison only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str, strip_punct: bool = False) -> str:
    t = text.lower()
    if strip_punct:
        t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_dedup(
    records: list[dict], strip_punct: bool = False
) -> tuple[list[dict], dict[str, list[str]]]:
    """Group by sha256(normalize(text)). Return (survivors, groups).

    survivors: one record per hash group — the first-seen row in input order.
    groups:    hash -> [doc_id, doc_id, ...] for every row that hashed there.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    first_seen: dict[str, dict] = {}
    for r in records:
        h = sha(normalize(r["text"], strip_punct=strip_punct))
        groups[h].append(r["doc_id"])
        first_seen.setdefault(h, r)
    return list(first_seen.values()), dict(groups)


def match_ground_truth(
    groups: dict[str, list[str]], ground_truth_path: Path
) -> tuple[dict[str, int], dict[str, int]]:
    """For each ground-truth pair, did both docs land in the same hash group?

    Returns (caught_by_edit_type, missed_by_edit_type).
    """
    id_to_hash: dict[str, str] = {}
    for h, ids in groups.items():
        for d in ids:
            id_to_hash[d] = h
    caught: dict[str, int] = defaultdict(int)
    missed: dict[str, int] = defaultdict(int)
    with ground_truth_path.open() as f:
        for row in csv.DictReader(f):
            a = id_to_hash.get(row["doc_id_1"])
            b = id_to_hash.get(row["doc_id_2"])
            if a is not None and a == b:
                caught[row["edit_type"]] += 1
            else:
                missed[row["edit_type"]] += 1
    return dict(caught), dict(missed)


def summarize(groups: dict[str, list[str]]) -> dict:
    sizes = [len(g) for g in groups.values()]
    return {
        "total_rows": sum(sizes),
        "distinct_groups": len(sizes),
        "rows_removed": sum(sizes) - len(sizes),
        "groups_with_duplicates": sum(1 for s in sizes if s > 1),
        "largest_group_size": max(sizes) if sizes else 0,
        "top_10_group_sizes": sorted(sizes, reverse=True)[:10],
    }


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_mode(
    records: list[dict],
    strip_punct: bool,
    label: str,
    stats_path: Path,
    survivors_path: Path | None,
    ground_truth_path: Path | None,
) -> dict:
    survivors, groups = exact_dedup(records, strip_punct=strip_punct)
    stats = summarize(groups)
    stats["mode"] = label
    stats["strip_punct"] = strip_punct
    if ground_truth_path is not None:
        caught, missed = match_ground_truth(groups, ground_truth_path)
        stats["ground_truth_caught"] = caught
        stats["ground_truth_missed"] = missed

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    if survivors_path is not None:
        survivors_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(survivors_path, survivors)

    print(f"--- {label}  (strip_punct={strip_punct}) ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  survivors -> {survivors_path}")
    print(f"  stats     -> {stats_path}")
    print()
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    p.add_argument("--input", default=str(root / "data/documents_100k.jsonl"))
    p.add_argument(
        "--ground-truth",
        default=str(root / "data/ground_truth_pairs_100k.csv"),
    )
    p.add_argument("--data-dir", default=str(root / "data"))
    p.add_argument("--outputs-dir", default=str(root / "outputs"))
    args = p.parse_args()

    input_path = Path(args.input)
    stem = input_path.stem  # "documents_100k"
    data_dir = Path(args.data_dir)
    outputs_dir = Path(args.outputs_dir)
    gt_path = Path(args.ground_truth) if Path(args.ground_truth).exists() else None

    records = load_jsonl(input_path)
    print(f"[preprocess] loaded {len(records)} records from {input_path}")
    if gt_path is None:
        print(f"[preprocess] ground truth not found — skipping ground-truth match")

    run_mode(
        records,
        strip_punct=False,
        label="conservative",
        stats_path=outputs_dir / f"exact_dedup_stats_{stem}_conservative.json",
        survivors_path=data_dir / f"{stem}_after_exact.jsonl",
        ground_truth_path=gt_path,
    )

    run_mode(
        records,
        strip_punct=True,
        label="aggressive",
        stats_path=outputs_dir / f"exact_dedup_stats_{stem}_aggressive.json",
        survivors_path=data_dir / f"{stem}_after_exact_aggressive.jsonl",
        ground_truth_path=gt_path,
    )


if __name__ == "__main__":
    main()
