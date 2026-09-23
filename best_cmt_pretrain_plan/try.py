# /home/yihaoyu/docker/bag/MOFTransformer_upgrade/MOFTransformer/try.py

import os
import torch
from sacred.config.config_summary import ConfigSummary

import moftransformer
from moftransformer.config import config as _config

# 允许加载 ConfigSummary（你之前遇到过 torch load 的安全限制）
torch.serialization.add_safe_globals([ConfigSummary])

# 稍微吃 TensorCore（不改 AMP，只让 float32 matmul 更快）
torch.set_float32_matmul_precision("medium")

# 减少碎片（可选）
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.9",
)

if __name__ == "__main__":
    # ===== 路径配置 =====
    root_dataset = os.environ.get(
        "ALIGNN_TRY_ROOT_DATASET",
        "/home/yihaoyu/docker/bag/MOFTransformer_upgrade/MOFTransformer/all_N2",
    )
    downstream = os.environ.get("ALIGNN_TRY_DOWNSTREAM", "N2")
    log_dir = os.environ.get("ALIGNN_TRY_LOG_DIR", root_dataset)
    load_path = os.environ.get(
        "ALIGNN_TRY_LOAD_PATH",
        "/home/yihaoyu/docker/bag/MOFTransformer_upgrade/MOFTransformer/hmof_pretrain_GGM_MPP_1pct/logs_alignn_pretrain/alignn_hmof_pretrain_1pct_seed0_from_/version_6/checkpoints/best.ckpt",
    )

    # ===== N2 标签归一化参数 =====
    # ⚠️ 对比实验时：mean/std 必须和 CGCNN 版保持一致
    mean = 0
    std = 1

    # ===== 训练超参数 =====
    max_epochs = int(os.environ.get("ALIGNN_TRY_MAX_EPOCHS", "80"))
    micro_batch = 1

    # 有效 batch（不增加显存）
    effective_batch = int(os.environ.get("ALIGNN_TRY_BATCH_SIZE", "4"))
    batch_size = effective_batch

    default_lr = "5e-5" if load_path else "2e-4"
    learning_rate = float(os.environ.get("ALIGNN_TRY_LR", default_lr))
    weight_decay = float(os.environ.get("ALIGNN_TRY_WEIGHT_DECAY", "1e-3"))
    drop_rate = float(os.environ.get("ALIGNN_TRY_DROP_RATE", "0.1"))
    freeze_backbone_epochs = int(os.environ.get("ALIGNN_TRY_FREEZE_BACKBONE_EPOCHS", "0"))

    # ===== 读取默认 config 并覆盖关键字段 =====
    config = _config()

    # 只做回归（避免其它任务 head 影响）
    config["loss_names"] = {
        "ggm": 0, "mpp": 0, "mtp": 0, "vfp": 0, "moc": 0, "bbc": 0,
        "classification": 0,
        "regression": 1,
    }

    config.update({
        "root_dataset": root_dataset,
        "downstream": downstream,
        "log_dir": log_dir,

        "max_epochs": max_epochs,

        # 有效 batch（用于步数/日志等）
        "batch_size": batch_size,
        # 实际喂给 GPU 的 micro batch
        "per_gpu_batchsize": micro_batch,

        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "drop_rate": drop_rate,

        "mean": mean,
        "std": std,

        # ===== 使用你刚训好的 ALIGNN 预训练 ckpt =====
        "load_path": load_path,
        "resume_from": None,    # 不从历史 ckpt 续训
        "test_only": False,

        # ===== 单卡单 GPU =====
        "accelerator": "gpu",
        "devices": 1,
        "num_nodes": 1,
        "precision": os.environ.get("ALIGNN_TRY_PRECISION", "16-mixed"),

        "num_workers": 4,

        # ===== ✅ 为了“同一条件”更稳：显式固定 seed（别依赖默认）=====
        "seed": 0,

        # ===== 可选：确保不走分布式策略（你之前也不想用 DDP）=====
        "strategy": None,

        # ===== 与预训练 ckpt 保持一致的模型结构 =====
        "img_size": int(os.environ.get("ALIGNN_TRY_IMG_SIZE", "20")),
        "hid_dim": int(os.environ.get("ALIGNN_TRY_HID_DIM", "384")),
        "num_layers": int(os.environ.get("ALIGNN_TRY_NUM_LAYERS", "6")),
        "alignn_layers": int(os.environ.get("ALIGNN_TRY_ALIGNN_LAYERS", "2")),
        "graph_dropout": float(os.environ.get("ALIGNN_TRY_GRAPH_DROPOUT", "0.1")),
        "freeze_backbone_epochs": freeze_backbone_epochs,
    })

    print("==== Final training config (key fields) ====")
    for k in [
        "root_dataset", "downstream",
        "max_epochs", "batch_size", "per_gpu_batchsize",
        "learning_rate", "weight_decay", "drop_rate",
        "mean", "std",
        "seed",
        "load_path", "resume_from", "test_only",
        "accelerator", "devices", "strategy",
        "precision", "num_workers",
        "img_size", "hid_dim", "num_layers", "alignn_layers",
        "freeze_backbone_epochs",
    ]:
        print(f"{k}: {config.get(k)}")

    print("\nCUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET"))
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))

    # 直接跑
    moftransformer.run(**config)

# 用法示例：
# export CUDA_VISIBLE_DEVICES=GPU-595536f6-a9e7-a60d-ab7e-c0341726e449
# python /home/yihaoyu/docker/bag/MOFTransformer_upgrade/MOFTransformer/try.py
