#!/usr/bin/env python3
"""
从 CIF 文件批量生成可直接用于预训练的化学/局部环境代理标签。

目标：
1. 不依赖人工标签；
2. 尽量只用当前环境已有依赖（pymatgen / ase）；
3. 输出统一的 csv + jsonl，后续可以很方便按 cif_id 接到 dataset。

当前输出分为三类：
- 全局组成/晶体统计：atom_count, density, volume, 元素计数等
- 局部化学环境代理：金属中心、配位数、杂原子比例
- 官能团近似标签：amine-like / hydroxyl-like / carboxylate-like / halogen / sulfur-like

说明：
- 这里的“官能团”是基于结构和近邻的启发式近似，不保证达到严格有机化学判定精度；
- 但它们足够用来做一版预训练代理任务，尤其适合先做方法验证。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from ase import neighborlist
from ase.io import read
from pymatgen.core import Element


SUPPORTED_SUFFIXES = {".cif"}
HALOGENS = {"F", "Cl", "Br", "I"}
DONOR_ELEMENTS = {"N", "O", "S", "P"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 CIF 批量生成化学/官能团代理标签。"
    )
    parser.add_argument(
        "--cif-dir",
        type=Path,
        required=True,
        help="包含 cif 文件的目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出目录，会生成 chem_labels.csv / chem_labels.jsonl / chem_summary.json。",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="是否递归扫描 cif 文件。",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="仅处理前 N 个文件，0 表示处理全部。",
    )
    return parser.parse_args()


def iter_cif_paths(root: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.cif" if recursive else "*.cif"
    paths = sorted(p for p in root.glob(pattern) if p.suffix.lower() in SUPPORTED_SUFFIXES)
    return paths


def build_neighbor_map(atoms) -> Dict[int, List[int]]:
    cutoffs = neighborlist.natural_cutoffs(atoms)
    nl = neighborlist.NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    neigh_map: Dict[int, List[int]] = {}
    for i in range(len(atoms)):
        indices, _ = nl.get_neighbors(i)
        neigh_map[i] = indices.tolist()
    return neigh_map


def safe_div(a: float, b: float) -> float:
    if abs(b) < 1e-12:
        return 0.0
    return float(a) / float(b)


def is_metal(symbol: str) -> bool:
    try:
        return Element(symbol).is_metal
    except Exception:
        return False


def count_neighbors_by_symbol(symbols: List[str], nbr_idx: List[int], target: str) -> int:
    return sum(1 for j in nbr_idx if symbols[j] == target)


def count_neighbors_in_set(symbols: List[str], nbr_idx: List[int], targets: set[str]) -> int:
    return sum(1 for j in nbr_idx if symbols[j] in targets)


def detect_proxy_labels(symbols: List[str], neigh_map: Dict[int, List[int]]) -> Dict[str, int]:
    has_halogen = int(any(s in HALOGENS for s in symbols))
    has_nitrogen = int("N" in symbols)
    has_oxygen = int("O" in symbols)
    has_sulfur = int("S" in symbols)
    has_phosphorus = int("P" in symbols)

    amine_like = 0
    hydroxyl_like = 0
    carboxylate_like = 0
    sulfur_group_like = 0
    open_metal_site_like = 0

    for i, s in enumerate(symbols):
        nbr_idx = neigh_map.get(i, [])

        if s == "N":
            n_c = count_neighbors_by_symbol(symbols, nbr_idx, "C")
            n_h = count_neighbors_by_symbol(symbols, nbr_idx, "H")
            deg = len(nbr_idx)
            if (n_c >= 1 and deg <= 3) or (n_c >= 1 and n_h >= 1):
                amine_like = 1

        elif s == "O":
            n_h = count_neighbors_by_symbol(symbols, nbr_idx, "H")
            n_c = count_neighbors_by_symbol(symbols, nbr_idx, "C")
            n_metal = sum(1 for j in nbr_idx if is_metal(symbols[j]))
            if n_h >= 1 or (n_c >= 1 and n_metal == 0 and len(nbr_idx) <= 2):
                hydroxyl_like = 1

        elif s == "C":
            n_o = count_neighbors_by_symbol(symbols, nbr_idx, "O")
            if n_o >= 2:
                carboxylate_like = 1

        elif s == "S":
            n_o = count_neighbors_by_symbol(symbols, nbr_idx, "O")
            n_c = count_neighbors_by_symbol(symbols, nbr_idx, "C")
            if n_o >= 1 or n_c >= 1:
                sulfur_group_like = 1

        if is_metal(s):
            deg = len(nbr_idx)
            donor_nbrs = count_neighbors_in_set(symbols, nbr_idx, DONOR_ELEMENTS)
            if deg <= 4 and donor_nbrs >= 2:
                open_metal_site_like = 1

    return {
        "has_halogen": has_halogen,
        "has_nitrogen": has_nitrogen,
        "has_oxygen": has_oxygen,
        "has_sulfur": has_sulfur,
        "has_phosphorus": has_phosphorus,
        "fg_amine_like": amine_like,
        "fg_hydroxyl_like": hydroxyl_like,
        "fg_carboxylate_like": carboxylate_like,
        "fg_sulfur_like": sulfur_group_like,
        "fg_open_metal_site_like": open_metal_site_like,
    }


def summarize_structure(cif_path: Path) -> Tuple[Dict[str, object], Dict[str, int]]:
    atoms = read(str(cif_path))
    symbols = atoms.get_chemical_symbols()
    counts = Counter(symbols)
    neigh_map = build_neighbor_map(atoms)

    atom_count = len(symbols)
    metal_count = sum(v for k, v in counts.items() if is_metal(k))
    nonmetal_count = atom_count - metal_count
    unique_elements = len(counts)
    density = float(atoms.get_masses().sum() / atoms.get_volume() / 0.602214076) if atoms.get_volume() > 1e-12 else 0.0
    volume = float(atoms.get_volume())

    coord_numbers = [len(neigh_map[i]) for i in range(atom_count)]
    avg_coord = sum(coord_numbers) / atom_count if atom_count else 0.0
    max_coord = max(coord_numbers) if coord_numbers else 0
    min_coord = min(coord_numbers) if coord_numbers else 0

    proxy_labels = detect_proxy_labels(symbols, neigh_map)

    row: Dict[str, object] = {
        "cif_id": cif_path.stem,
        "cif_path": str(cif_path),
        "atom_count": atom_count,
        "unique_elements": unique_elements,
        "metal_count": metal_count,
        "nonmetal_count": nonmetal_count,
        "metal_fraction": safe_div(metal_count, atom_count),
        "volume": volume,
        "density": density,
        "avg_coordination": float(avg_coord),
        "min_coordination": int(min_coord),
        "max_coordination": int(max_coord),
        "count_H": counts.get("H", 0),
        "count_C": counts.get("C", 0),
        "count_N": counts.get("N", 0),
        "count_O": counts.get("O", 0),
        "count_S": counts.get("S", 0),
        "count_P": counts.get("P", 0),
        "count_F": counts.get("F", 0),
        "count_Cl": counts.get("Cl", 0),
        "count_Br": counts.get("Br", 0),
        "count_I": counts.get("I", 0),
        "fraction_C": safe_div(counts.get("C", 0), atom_count),
        "fraction_N": safe_div(counts.get("N", 0), atom_count),
        "fraction_O": safe_div(counts.get("O", 0), atom_count),
        "fraction_metal": safe_div(metal_count, atom_count),
    }
    row.update(proxy_labels)
    return row, counts


def main() -> None:
    args = parse_args()
    cif_paths = iter_cif_paths(args.cif_dir, args.recursive)
    if args.max_files > 0:
        cif_paths = cif_paths[: args.max_files]
    if not cif_paths:
        raise SystemExit(f"没有在 {args.cif_dir} 下找到 cif 文件。")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "chem_labels.csv"
    jsonl_path = args.output_dir / "chem_labels.jsonl"
    summary_path = args.output_dir / "chem_summary.json"

    rows: List[Dict[str, object]] = []
    element_counter: Counter[str] = Counter()
    total = len(cif_paths)
    start_time = time.time()

    print(f"开始生成化学/官能团代理标签: total={total}, source={args.cif_dir}", flush=True)

    for idx, path in enumerate(cif_paths, start=1):
        row, counts = summarize_structure(path)
        rows.append(row)
        element_counter.update(counts)

        if idx <= 3 or idx % 50 == 0 or idx == total:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0.0
            eta = (total - idx) / rate if rate > 0 else 0.0
            print(
                f"[chem] {idx}/{total} done | current={path.name} | elapsed={elapsed/60:.1f}m | rate={rate:.2f} cif/s | eta={eta/60:.1f}m",
                flush=True,
            )

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "num_cifs": len(rows),
        "source_dir": str(args.cif_dir),
        "output_dir": str(args.output_dir),
        "top_elements": element_counter.most_common(20),
        "label_positive_counts": {
            k: int(sum(int(row[k]) for row in rows))
            for k in rows[0].keys()
            if k.startswith("fg_") or k.startswith("has_")
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"已生成: {csv_path}")
    print(f"已生成: {jsonl_path}")
    print(f"已生成: {summary_path}")
    print(f"样本数: {len(rows)}")


if __name__ == "__main__":
    main()
