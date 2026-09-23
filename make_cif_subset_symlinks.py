import math
import os
import random
from pathlib import Path

from tqdm import tqdm


SOURCE_DIR = Path(os.environ.get("CIF_SUBSET_SOURCE_DIR", "/home/yihaoyu/dataset/primitive_cif"))
TARGET_DIR = Path(
    os.environ.get(
        "CIF_SUBSET_TARGET_DIR",
        "/home/yihaoyu/docker/bag/MOFTransformer_upgrade/data_subsets/primitive_cif_1pct_seed42",
    )
)
FRACTION = float(os.environ.get("CIF_SUBSET_FRACTION", "0.01"))
SEED = int(os.environ.get("CIF_SUBSET_SEED", "42"))


def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"source dir does not exist: {SOURCE_DIR}")
    if not (0 < FRACTION <= 1):
        raise ValueError(f"fraction must be in (0, 1], got {FRACTION}")

    cif_paths = sorted(SOURCE_DIR.glob("*.cif"))
    total = len(cif_paths)
    if total == 0:
        raise ValueError(f"no cif files found in: {SOURCE_DIR}")

    sample_size = max(1, math.ceil(total * FRACTION))
    rng = random.Random(SEED)
    sampled_paths = sorted(rng.sample(cif_paths, sample_size), key=lambda p: p.name)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"source_dir={SOURCE_DIR}")
    print(f"target_dir={TARGET_DIR}")
    print(f"fraction={FRACTION}")
    print(f"seed={SEED}")
    print(f"total_cifs={total}")
    print(f"sample_size={sample_size}")

    created = 0
    skipped = 0
    for src in tqdm(sampled_paths, desc="Creating subset symlinks", unit="cif"):
        dest = TARGET_DIR / src.name
        if dest.exists() or dest.is_symlink():
            skipped += 1
            continue
        dest.symlink_to(src)
        created += 1

    print(f"created_symlinks={created}")
    print(f"skipped_existing={skipped}")
    print("subset ready")


if __name__ == "__main__":
    main()
