import argparse
import os
from pathlib import Path

import torch
from sacred.config.config_summary import ConfigSummary

import moftransformer
from moftransformer.config import config as _config


torch.serialization.add_safe_globals([ConfigSummary])
torch.set_float32_matmul_precision("medium")

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.9",
)


PRESET_TASKS = {
    "mtp_bbc_vfp": ["mtp", "bbc", "vfp"],
    "full": ["ggm", "mpp", "mtp", "vfp", "moc", "bbc"],
}

ALL_LOSSES = [
    "ggm",
    "mpp",
    "mtp",
    "vfp",
    "moc",
    "bbc",
    "classification",
    "regression",
]


# ===== Edit These Settings At The Top =====
ROOT_DATASET = "/path/to/your/pretrain_dataset"
LOG_DIR = None
EXP_NAME = "pretrained_mof_alignn"

# Two ways to choose tasks:
# 1) Set PRESET to "mtp_bbc_vfp" or "full"
# 2) Set TASKS manually, e.g. ["mtp", "bbc", "vfp"], then PRESET is ignored
PRESET = "mtp_bbc_vfp"
TASKS = None

MAX_EPOCHS = 100
BATCH_SIZE = 32
PER_GPU_BATCHSIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
SEED = 0
NUM_WORKERS = 4

ACCELERATOR = "gpu"
DEVICES = 1
NUM_NODES = 1
PRECISION = 32
STRATEGY = None

LOAD_PATH = ""
RESUME_FROM = None
VAL_CHECK_INTERVAL = 1.0
IMG_SIZE = 30
NBR_FEA_LEN = 64


def build_loss_names(tasks):
    loss_names = {name: 0 for name in ALL_LOSSES}
    for task in tasks:
        if task not in loss_names:
            raise ValueError(f"Unsupported pretraining task: {task}")
        loss_names[task] = 1
    return loss_names


def validate_pretrain_dataset(root_dataset: Path, tasks):
    required = [
        root_dataset / "train.json",
        root_dataset / "val.json",
        root_dataset / "test.json",
    ]

    for split in ["train", "val", "test"]:
        split_dir = root_dataset / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

    extra_json_tasks = {"mtp", "vfp", "moc", "bbc"}
    for task in tasks:
        if task in extra_json_tasks:
            required.extend(
                [
                    root_dataset / f"train_{task}.json",
                    root_dataset / f"val_{task}.json",
                    root_dataset / f"test_{task}.json",
                ]
            )

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Missing required pretraining files:\n{joined}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Explicit entrypoint for MOFTransformer pretraining.",
    )
    parser.add_argument(
        "--root_dataset",
        default=None,
        help="Pretraining dataset root. Must contain train/ val/ test folders and train.json style files.",
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Directory for TensorBoard logs and checkpoints. Defaults to <root_dataset>/pretrain_logs.",
    )
    parser.add_argument(
        "--exp_name",
        default=None,
        help="Experiment name.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_TASKS.keys()),
        default=None,
        help="Native pretraining task preset from this repo.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional explicit task list. Overrides --preset. Supported: ggm mpp mtp vfp moc bbc",
    )
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--per_gpu_batchsize", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--load_path", default=None)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--val_check_interval", type=float, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--nbr_fea_len", type=int, default=None)
    return parser.parse_args()


def normalize_devices(value):
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    return int(text)


if __name__ == "__main__":
    args = parse_args()

    root_dataset_value = args.root_dataset if args.root_dataset is not None else ROOT_DATASET
    if not root_dataset_value or root_dataset_value == "/path/to/your/pretrain_dataset":
        raise ValueError(
            "Please edit ROOT_DATASET at the top of pretrain_entry.py "
            "or pass --root_dataset from the terminal."
        )

    root_dataset = Path(root_dataset_value).expanduser().resolve()
    log_dir = (
        Path(args.log_dir).expanduser().resolve()
        if args.log_dir
        else (
            Path(LOG_DIR).expanduser().resolve()
            if LOG_DIR
            else root_dataset / "pretrain_logs"
        )
    )

    preset = args.preset if args.preset is not None else PRESET
    tasks = args.tasks if args.tasks else (TASKS if TASKS else PRESET_TASKS[preset])
    validate_pretrain_dataset(root_dataset, tasks)

    config = _config()
    config["loss_names"] = build_loss_names(tasks)

    draw_false_grid = "ggm" in tasks

    config.update(
        {
            "root_dataset": str(root_dataset),
            "downstream": "",
            "log_dir": str(log_dir),
            "exp_name": args.exp_name if args.exp_name is not None else EXP_NAME,
            "max_epochs": args.max_epochs if args.max_epochs is not None else MAX_EPOCHS,
            "batch_size": args.batch_size if args.batch_size is not None else BATCH_SIZE,
            "per_gpu_batchsize": args.per_gpu_batchsize if args.per_gpu_batchsize is not None else PER_GPU_BATCHSIZE,
            "learning_rate": args.learning_rate if args.learning_rate is not None else LEARNING_RATE,
            "weight_decay": args.weight_decay if args.weight_decay is not None else WEIGHT_DECAY,
            "mean": None,
            "std": None,
            "load_path": args.load_path if args.load_path is not None else LOAD_PATH,
            "resume_from": args.resume_from if args.resume_from is not None else RESUME_FROM,
            "test_only": False,
            "accelerator": args.accelerator if args.accelerator is not None else ACCELERATOR,
            "devices": normalize_devices(args.devices if args.devices is not None else DEVICES),
            "num_nodes": args.num_nodes if args.num_nodes is not None else NUM_NODES,
            "precision": (
                int(args.precision) if str(args.precision).isdigit()
                else args.precision
            ) if args.precision is not None else PRECISION,
            "num_workers": args.num_workers if args.num_workers is not None else NUM_WORKERS,
            "seed": args.seed if args.seed is not None else SEED,
            "strategy": args.strategy if args.strategy is not None else STRATEGY,
            "draw_false_grid": draw_false_grid,
            "img_size": args.img_size if args.img_size is not None else IMG_SIZE,
            "nbr_fea_len": args.nbr_fea_len if args.nbr_fea_len is not None else NBR_FEA_LEN,
            "val_check_interval": args.val_check_interval if args.val_check_interval is not None else VAL_CHECK_INTERVAL,
        }
    )

    print("==== Pretraining config (key fields) ====")
    for key in [
        "root_dataset",
        "log_dir",
        "exp_name",
        "loss_names",
        "draw_false_grid",
        "max_epochs",
        "batch_size",
        "per_gpu_batchsize",
        "learning_rate",
        "weight_decay",
        "seed",
        "load_path",
        "resume_from",
        "accelerator",
        "devices",
        "strategy",
        "precision",
        "num_workers",
    ]:
        print(f"{key}: {config.get(key)}")

    print("\nCUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET"))
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))

    moftransformer.run(**config)
