"""Import calls.jsonl rows + response-shard lines from another run's raw export.

Stamps "prompt_variant" (immediately after "addressing_mode") on each imported
row and appends matching lines from that model's response shard. Idempotent:
rows/lines whose call_id already exists in the destination are skipped.

Usage:
    uv run python scripts/import_calls.py --from-raw /path/to/results/raw --variant segmentation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEST_RAW = Path(__file__).resolve().parents[1] / "results" / "raw"


def _safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def _load_call_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.is_file():
        return ids
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["call_id"])
    return ids


def _insert_prompt_variant(row: dict, variant: str) -> dict:
    out = {}
    for key, value in row.items():
        out[key] = value
        if key == "addressing_mode":
            out["prompt_variant"] = variant
    return out


def import_calls(from_raw: Path, variant: str, dest_raw: Path = DEST_RAW) -> tuple[int, dict[str, int]]:
    """Append new rows/shard lines from from_raw into dest_raw. Returns
    (rows appended, {shard_filename: lines appended})."""
    dest_calls = dest_raw / "calls.jsonl"
    existing_ids = _load_call_ids(dest_calls)

    new_rows: list[dict] = []
    new_ids_by_model: dict[str, set[str]] = {}
    with (from_raw / "calls.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            call_id = row["call_id"]
            if call_id in existing_ids:
                continue
            if "addressing_mode" not in row:
                raise ValueError(f"{call_id}: row has no addressing_mode field")
            new_rows.append(_insert_prompt_variant(row, variant))
            new_ids_by_model.setdefault(row["model"], set()).add(call_id)

    if new_rows:
        dest_calls.parent.mkdir(parents=True, exist_ok=True)
        with dest_calls.open("a", encoding="utf-8") as f:
            for row in new_rows:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")

    shard_counts = _import_shards(from_raw, dest_raw, new_ids_by_model)
    return len(new_rows), shard_counts


def _import_shards(from_raw: Path, dest_raw: Path, new_ids_by_model: dict[str, set[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for model, call_ids in new_ids_by_model.items():
        shard_name = f"{_safe_model_id(model)}.jsonl"
        src_shard = from_raw / "responses" / shard_name
        if not src_shard.is_file():
            raise FileNotFoundError(f"missing source shard for model {model!r}: {src_shard}")
        dest_shard = dest_raw / "responses" / shard_name
        existing_shard_ids = _load_call_ids(dest_shard)

        appended_lines: list[str] = []
        with src_shard.open(encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                rec = json.loads(stripped)
                cid = rec["call_id"]
                if cid in call_ids and cid not in existing_shard_ids:
                    appended_lines.append(stripped)

        if appended_lines:
            dest_shard.parent.mkdir(parents=True, exist_ok=True)
            with dest_shard.open("a", encoding="utf-8") as f:
                for line in appended_lines:
                    f.write(line + "\n")
        counts[shard_name] = len(appended_lines)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-raw", required=True, type=Path, help="source results/raw dir to import from")
    parser.add_argument("--variant", required=True, help="prompt_variant value to stamp on imported rows")
    args = parser.parse_args()

    rows_added, shard_counts = import_calls(args.from_raw, args.variant)
    print(f"calls.jsonl: {rows_added} row(s) appended")
    for shard, count in sorted(shard_counts.items()):
        print(f"responses/{shard}: {count} line(s) appended")


if __name__ == "__main__":
    main()
