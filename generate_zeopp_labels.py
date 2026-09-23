#!/usr/bin/env python3
"""
用 Zeo++ 批量计算孔结构描述符，并整理成 csv/jsonl。

说明：
- 该脚本是对 Zeo++ 的薄包装；
- 机器上当前未检测到 `network` 可执行文件，所以运行时需要显式提供 `--zeo-binary`；
- 输出字段以 N2 吸附较相关的孔结构描述符为主。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List


FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行 Zeo++ 并生成孔结构标签。")
    parser.add_argument("--cif-dir", type=Path, required=True, help="cif 文件目录。")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录。")
    parser.add_argument(
        "--zeo-binary",
        type=Path,
        default=None,
        help="Zeo++ 的 network 可执行文件路径。",
    )
    parser.add_argument("--recursive", action="store_true", help="递归扫描 cif。")
    parser.add_argument("--max-files", type=int, default=0, help="只处理前 N 个。")
    parser.add_argument(
        "--probe-radius",
        type=float,
        default=1.86,
        help="探针半径，N2 常用近似可以先从 1.86 开始试。",
    )
    parser.add_argument(
        "--channel-radius",
        type=float,
        default=1.86,
        help="channel 半径参数。",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50000,
        help="Zeo++ 采样数。",
    )
    return parser.parse_args()


def resolve_zeo_binary(path: Path | None) -> str:
    if path is not None and path.exists():
        return str(path)
    auto = shutil.which("network")
    if auto:
        return auto
    raise SystemExit(
        "没有找到 Zeo++ 可执行文件。请通过 --zeo-binary 显式传入 network 路径。"
    )


def iter_cif_paths(root: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.cif" if recursive else "*.cif"
    return sorted(root.glob(pattern))


def parse_all_floats(text: str) -> List[float]:
    return [float(x) for x in FLOAT_RE.findall(text)]


def parse_zeo_outputs(res_text: str, sa_text: str, vol_text: str) -> Dict[str, float | str | None]:
    result: Dict[str, float | str | None] = {
        "lcd": None,
        "pld": None,
        "lfpd": None,
        "asa_m2_cm3": None,
        "asa_m2_g": None,
        "nasa_m2_cm3": None,
        "nasa_m2_g": None,
        "av_cm3_g": None,
        "av_fraction": None,
        "nav_cm3_g": None,
        "nav_fraction": None,
        "raw_res": res_text.strip(),
        "raw_sa": sa_text.strip(),
        "raw_vol": vol_text.strip(),
    }

    res_vals = parse_all_floats(res_text)
    if len(res_vals) >= 3:
        result["pld"] = res_vals[0]
        result["lcd"] = res_vals[1]
        result["lfpd"] = res_vals[2]

    sa_vals = parse_all_floats(sa_text)
    if len(sa_vals) >= 4:
        result["asa_m2_cm3"] = sa_vals[0]
        result["asa_m2_g"] = sa_vals[1]
        result["nasa_m2_cm3"] = sa_vals[2]
        result["nasa_m2_g"] = sa_vals[3]

    vol_vals = parse_all_floats(vol_text)
    if len(vol_vals) >= 4:
        result["av_cm3_g"] = vol_vals[0]
        result["av_fraction"] = vol_vals[1]
        result["nav_cm3_g"] = vol_vals[2]
        result["nav_fraction"] = vol_vals[3]

    return result


def run_zeopp(zeo_binary: str, cif_path: Path, workdir: Path, probe: float, channel: float, num_samples: int) -> Dict[str, float | str | None]:
    stem = cif_path.stem
    res_path = workdir / f"{stem}.res"
    sa_path = workdir / f"{stem}.sa"
    vol_path = workdir / f"{stem}.vol"

    subprocess.run(
        [
            zeo_binary,
            "-ha",
            "-res",
            str(res_path),
            "-sa",
            str(probe),
            str(channel),
            str(num_samples),
            str(sa_path),
            "-vol",
            str(probe),
            str(channel),
            str(num_samples),
            str(vol_path),
            str(cif_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    res_text = res_path.read_text(encoding="utf-8", errors="ignore") if res_path.exists() else ""
    sa_text = sa_path.read_text(encoding="utf-8", errors="ignore") if sa_path.exists() else ""
    vol_text = vol_path.read_text(encoding="utf-8", errors="ignore") if vol_path.exists() else ""
    return parse_zeo_outputs(res_text, sa_text, vol_text)


def main() -> None:
    args = parse_args()
    zeo_binary = resolve_zeo_binary(args.zeo_binary)
    cif_paths = iter_cif_paths(args.cif_dir, args.recursive)
    if args.max_files > 0:
        cif_paths = cif_paths[: args.max_files]
    if not cif_paths:
        raise SystemExit(f"没有在 {args.cif_dir} 下找到 cif 文件。")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "zeopp_labels.csv"
    jsonl_path = args.output_dir / "zeopp_labels.jsonl"
    summary_path = args.output_dir / "zeopp_summary.json"

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    total = len(cif_paths)
    start_time = time.time()

    print(
        f"开始生成 Zeo++ 孔结构标签: total={total}, source={args.cif_dir}, probe={args.probe_radius}, samples={args.num_samples}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="zeopp_labels_") as tmp:
        workdir = Path(tmp)
        for idx, cif_path in enumerate(cif_paths, start=1):
            try:
                parsed = run_zeopp(
                    zeo_binary=zeo_binary,
                    cif_path=cif_path,
                    workdir=workdir,
                    probe=args.probe_radius,
                    channel=args.channel_radius,
                    num_samples=args.num_samples,
                )
                row = {
                    "cif_id": cif_path.stem,
                    "cif_path": str(cif_path),
                    **parsed,
                }
                rows.append(row)
                status = 'ok'
            except Exception as exc:
                failures.append({"cif_id": cif_path.stem, "error": str(exc)})
                status = f'fail: {exc}'

            if idx <= 3 or idx % 10 == 0 or idx == total:
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0.0
                eta = (total - idx) / rate if rate > 0 else 0.0
                print(
                    f"[zeopp] {idx}/{total} done | current={cif_path.name} | status={status} | success={len(rows)} | failed={len(failures)} | elapsed={elapsed/60:.1f}m | rate={rate:.2f} cif/s | eta={eta/60:.1f}m",
                    flush=True,
                )

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "num_success": len(rows),
        "num_failed": len(failures),
        "source_dir": str(args.cif_dir),
        "output_dir": str(args.output_dir),
        "zeo_binary": zeo_binary,
        "probe_radius": args.probe_radius,
        "channel_radius": args.channel_radius,
        "num_samples": args.num_samples,
        "failures": failures[:200],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if rows:
        print(f"已生成: {csv_path}")
        print(f"已生成: {jsonl_path}")
    print(f"已生成: {summary_path}")
    print(f"成功: {len(rows)} | 失败: {len(failures)}")


if __name__ == "__main__":
    main()
