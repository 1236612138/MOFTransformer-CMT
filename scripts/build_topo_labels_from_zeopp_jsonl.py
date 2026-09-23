#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


CHANNEL_RE = re.compile(r"Number_of_channels:\s*(\d+)")
POCKET_RE = re.compile(r"Number_of_pockets:\s*(\d+)")


def extract_int(pattern: re.Pattern[str], text: str, default: int = 0) -> int:
    match = pattern.search(text or "")
    return int(match.group(1)) if match else default


def main() -> None:
    root = Path("/home/yihaoyu/docker/bag/MOFTransformer_CMT")
    src = root / "generated_labels/hmof_1pct_zeopp/zeopp_labels.jsonl"
    out_dir = root / "generated_labels/hmof_1pct_topo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "topo_labels.csv"

    fieldnames = [
        "cif_id",
        "num_channels",
        "num_pockets",
        "has_accessible_channel",
        "has_pocket",
    ]

    rows = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            raw_sa = obj.get("raw_sa", "")
            raw_vol = obj.get("raw_vol", "")
            num_channels = extract_int(CHANNEL_RE, raw_sa) or extract_int(CHANNEL_RE, raw_vol)
            num_pockets = extract_int(POCKET_RE, raw_sa) or extract_int(POCKET_RE, raw_vol)
            rows.append(
                {
                    "cif_id": obj["cif_id"],
                    "num_channels": float(num_channels),
                    "num_pockets": float(num_pockets),
                    "has_accessible_channel": float(num_channels > 0),
                    "has_pocket": float(num_pockets > 0),
                }
            )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
