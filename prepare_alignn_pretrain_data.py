import json
import os
import random
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "/home/yihaoyu/docker/bag/MOFTransformer_upgrade/MOFTransformer")

from moftransformer.utils import prepare_data
from tqdm import tqdm

try:
    ENCODING_WARNING = EncodingWarning
except NameError:
    ENCODING_WARNING = None


# 只包含原始 CIF 的目录。
ROOT_CIFS = Path(
    os.environ.get("ALIGNN_PREP_ROOT_CIFS", "/home/yihaoyu/dataset/primitive_cif")
)

# 纯预训练数据输出目录。
ROOT_DATASET = Path(
    os.environ.get(
        "ALIGNN_PREP_ROOT_DATASET",
        "/home/yihaoyu/docker/bag/MOFTransformer_upgrade/MOFTransformer/hmof_pretrain_GGM_MPP",
    )
)

# 数据划分比例。
TRAIN_FRACTION = 0.98
TEST_FRACTION = 0.01
SEED = 42
NUM_WORKERS = int(os.environ.get("ALIGNN_PREP_NUM_WORKERS", "24"))
CHUNKSIZE = int(os.environ.get("ALIGNN_PREP_CHUNKSIZE", "16"))
PROGRESS_EVERY = int(os.environ.get("ALIGNN_PREP_PROGRESS_EVERY", "100"))

# 纯预训练不需要真实监督标签，这里统一写占位值。
PLACEHOLDER_TARGET = 0.0

# 图和能量网格构造参数。
PREPARE_KWARGS = {
    "seed": SEED,
    "train_fraction": TRAIN_FRACTION,
    "test_fraction": TEST_FRACTION,
    "num_workers": NUM_WORKERS,
    "chunksize": CHUNKSIZE,
    "progress_every": PROGRESS_EVERY,
    "max_num_nbr": 12,
    "num_angle_basis": 8,
    "max_length": 60.0,
    "min_length": 30.0,
    # 你的 CIF 已经是 primitive_cif，图侧继续使用 primitive 更合适。
    "get_primitive": True,
    "max_num_unique_atoms": 300,
}


def build_placeholder_json(root_dataset: Path, placeholder_target: float) -> None:
    for split in tqdm(
        ["train", "val", "test"],
        desc="Building split JSON",
        unit="split",
    ):
        split_dir = root_dataset / split
        if not split_dir.exists():
            raise FileNotFoundError(f"split directory does not exist: {split_dir}")

        cif_paths = sorted(split_dir.glob("*.cif"))
        cif_ids = []
        for path in tqdm(
            cif_paths,
            desc=f"Collecting {split} CIF ids",
            unit="cif",
            leave=False,
        ):
            cif_ids.append(path.stem)

        if not cif_ids:
            raise ValueError(f"no cif files found in split directory: {split_dir}")

        payload = {}
        for cif_id in tqdm(
            cif_ids,
            desc=f"Writing {split}.json payload",
            unit="entry",
            leave=False,
        ):
            payload[cif_id] = placeholder_target

        out_path = root_dataset / f"{split}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f)

        print(f"wrote {out_path} with {len(payload)} entries")


def print_summary(root_dataset: Path) -> None:
    for split in ["train", "val", "test"]:
        split_dir = root_dataset / split
        cif_count = sum(1 for _ in split_dir.glob("*.cif"))
        graph_count = sum(1 for _ in split_dir.glob("*.graphdata"))
        grid_count = sum(1 for _ in split_dir.glob("*.grid"))
        griddata_count = sum(1 for _ in split_dir.glob("*.griddata16"))
        print(
            f"{split}: cif={cif_count}, graphdata={graph_count}, "
            f"grid={grid_count}, griddata16={griddata_count}"
        )


def main() -> None:
    if not ROOT_CIFS.exists():
        raise FileNotFoundError(f"ROOT_CIFS does not exist: {ROOT_CIFS}")

    ROOT_DATASET.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    if ENCODING_WARNING is not None:
        warnings.filterwarnings("ignore", category=ENCODING_WARNING)
    warnings.filterwarnings(
        "ignore",
        message=".*get_structures is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Issues encountered while parsing CIF.*",
        category=UserWarning,
    )

    print("starting pure-CIF ALIGNN pretrain data preparation")
    print(f"root_cifs={ROOT_CIFS}")
    print(f"root_dataset={ROOT_DATASET}")
    print(f"prepare_kwargs={PREPARE_KWARGS}")

    # 传空列表，跳过 prepare_data 里的下游任务 json 切分逻辑。
    prepare_data(
        root_cifs=str(ROOT_CIFS),
        root_dataset=str(ROOT_DATASET),
        downstream=[],
        **PREPARE_KWARGS,
    )

    # 为纯预训练补齐 train.json / val.json / test.json。
    build_placeholder_json(ROOT_DATASET, PLACEHOLDER_TARGET)
    print_summary(ROOT_DATASET)

    print("\npretrain dataset is ready")
    print("use this root_dataset in pretrain_alignn_hmof.py:")
    print(ROOT_DATASET)


if __name__ == "__main__":
    main()
