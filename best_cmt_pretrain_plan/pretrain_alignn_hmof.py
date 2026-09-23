# 这个脚本是一个“ALIGNN 版 MOFTransformer 的 hMOF 预训练入口”。
#
# 设计目标：
# 1. 不改你现有训练主干，只补一个清晰的启动脚本。
# 2. 默认按“预训练”思路跑，而不是下游回归微调。
# 3. 让你在脚本顶部就能改路径、任务开关、显卡和显存策略。
# 4. 每一行都尽量写注释，方便你后续自己改。

# 导入 os，用来设置环境变量和检查路径。
import os

# 导入 pathlib.Path，用来更稳地拼接和检查文件路径。
from pathlib import Path

# 导入 torch，用来做 CUDA 可用性检查和 safe globals 注册。
import torch

# 导入 Sacred 的 ConfigSummary，避免旧 ckpt / 配置对象反序列化时报安全限制错误。
from sacred.config.config_summary import ConfigSummary

# 导入项目主包，后面直接调用 moftransformer.run(**config)。
import moftransformer

# 导入项目默认配置函数，先拿一份基础配置再按预训练需求覆盖。
from moftransformer.config import config as _config


# 允许 torch.load 安全反序列化 ConfigSummary。
torch.serialization.add_safe_globals([ConfigSummary])

# 让 float32 matmul 在支持 Tensor Core 的卡上更快一些。
torch.set_float32_matmul_precision("medium")

# 设置 CUDA 内存分配策略，减少碎片，并允许可扩展 segment。
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.9",
)


# 这个函数只负责检查“你打开了哪些预训练任务”，并返回 loss_names 字典。
def build_loss_names():
    # 这里定义每个可选任务是否开启。
    #
    # 说明：
    # ggm = Graph Grid Matching，只需要 false_grid，不需要额外标签 json。
    # mpp = Masked Patch Prediction，只需要 grid，不需要额外标签 json。
    # mtp = MOF Topology Prediction，需要 train_mtp.json / val_mtp.json / test_mtp.json。
    # vfp = Void Fraction Prediction，需要 train_vfp.json / val_vfp.json / test_vfp.json。
    # moc = Metal Organic Classification，需要 train_moc.json / val_moc.json / test_moc.json。
    # bbc = Building Block Classification，需要 train_bbc.json / val_bbc.json / test_bbc.json。
    #
    # 这里不再只支持开关，而是支持“任务权重”。
    # 比如：ggm=0.2, mpp=1.0，表示保留少量 ggm 约束，但把主要优化压力放在 mpp。
    def env_weight(name: str, default: str) -> float:
        value = float(os.environ.get(name, default).strip())
        if value < 0:
            raise ValueError(f"{name} 不能小于 0，当前值为 {value}")
        return value

    ggm_weight = env_weight("ALIGNN_PRETRAIN_GGM_WEIGHT", "1")
    mpp_weight = env_weight("ALIGNN_PRETRAIN_MPP_WEIGHT", "1")
    mtp_weight = env_weight("ALIGNN_PRETRAIN_MTP_WEIGHT", "0")
    vfp_weight = env_weight("ALIGNN_PRETRAIN_VFP_WEIGHT", "0")
    moc_weight = env_weight("ALIGNN_PRETRAIN_MOC_WEIGHT", "0")
    bbc_weight = env_weight("ALIGNN_PRETRAIN_BBC_WEIGHT", "0")

    # 组装项目内部要求的 loss_names 格式。
    loss_names = {
        "ggm": ggm_weight,
        "mpp": mpp_weight,
        "mtp": mtp_weight,
        "vfp": vfp_weight,
        "moc": moc_weight,
        "bbc": bbc_weight,
        "classification": 0,
        "regression": 0,
    }

    # 防止你所有预训练任务都关掉，结果脚本变成“什么都不学”。
    if sum(loss_names.values()) == 0:
        raise ValueError("至少要开启一个预训练任务，例如 ggm 或 mpp。")

    # 返回最终任务配置。
    return loss_names


# 这个函数检查数据目录里必须存在哪些文件，避免训练跑到一半才报错。
def validate_pretrain_dataset(root_dataset: Path, loss_names: dict):
    # 定义最基础必须存在的 split json。
    required_files = [
        root_dataset / "train.json",
        root_dataset / "val.json",
        root_dataset / "test.json",
    ]

    # 如果开启了 mtp，就要求对应标签文件存在。
    if loss_names["mtp"] > 0:
        required_files.extend(
            [
                root_dataset / "train_mtp.json",
                root_dataset / "val_mtp.json",
                root_dataset / "test_mtp.json",
            ]
        )

    # 如果开启了 vfp，就要求对应标签文件存在。
    if loss_names["vfp"] > 0:
        required_files.extend(
            [
                root_dataset / "train_vfp.json",
                root_dataset / "val_vfp.json",
                root_dataset / "test_vfp.json",
            ]
        )

    # 如果开启了 moc，就要求对应标签文件存在。
    if loss_names["moc"] > 0:
        required_files.extend(
            [
                root_dataset / "train_moc.json",
                root_dataset / "val_moc.json",
                root_dataset / "test_moc.json",
            ]
        )

    # 如果开启了 bbc，就要求对应标签文件存在。
    if loss_names["bbc"] > 0:
        required_files.extend(
            [
                root_dataset / "train_bbc.json",
                root_dataset / "val_bbc.json",
                root_dataset / "test_bbc.json",
            ]
        )

    # 收集缺失文件，统一报错，方便你一次补齐。
    missing_files = [str(path) for path in required_files if not path.exists()]

    # 只要缺文件，就直接停止并明确告诉你缺了什么。
    if missing_files:
        raise FileNotFoundError(
            "hMOF 预训练数据不完整，缺少这些文件：\n" + "\n".join(missing_files)
        )


# 主入口。
if __name__ == "__main__":
    # =========================
    # 1. 你最常改的路径和实验名
    # =========================

    # 这里改成你的 hMOF 预训练数据目录。
    #
    # 注意：
    # 这是“预训练数据目录”，里面应当是 train.json / val.json / test.json，
    # 而不是 train_N2.json 这种下游回归数据命名。
    root_dataset = Path(
        os.environ.get(
            "ALIGNN_PRETRAIN_ROOT_DATASET",
            "/home/yihaoyu/docker/bag/MOFTransformer_upgrade/MOFTransformer/hmof_pretrain",
        )
    )

    # 这里写日志输出目录。
    log_dir = Path(
        os.environ.get(
            "ALIGNN_PRETRAIN_LOG_DIR",
            str(root_dataset / "logs_alignn_pretrain"),
        )
    )

    # 这里写实验名，后面 TensorBoard 和 checkpoint 目录会用到。
    exp_name = os.environ.get("ALIGNN_PRETRAIN_EXP_NAME", "alignn_hmof_pretrain")

    # 如果要做第二阶段续训，可以在这里手动指定上一阶段 ckpt。
    load_path = os.environ.get("ALIGNN_PRETRAIN_LOAD_PATH", "").strip()

    # 预训练时不需要下游任务名，所以这里必须留空字符串。
    downstream = ""

    # =========================
    # 2. 训练超参数
    # =========================

    # 训练轮数。
    max_epochs = int(os.environ.get("ALIGNN_PRETRAIN_MAX_EPOCHS", "50"))

    # 单卡实际 micro batch。
    per_gpu_batchsize = int(os.environ.get("ALIGNN_PRETRAIN_PER_GPU_BATCHSIZE", "1"))

    # 有效 batch，用于梯度累积。
    batch_size = int(os.environ.get("ALIGNN_PRETRAIN_BATCH_SIZE", "4"))

    # 学习率。
    learning_rate = float(os.environ.get("ALIGNN_PRETRAIN_LR", "1e-4"))

    # 权重衰减。
    weight_decay = float(os.environ.get("ALIGNN_PRETRAIN_WEIGHT_DECAY", "1e-2"))

    # dropout。
    drop_rate = float(os.environ.get("ALIGNN_PRETRAIN_DROP_RATE", "0.1"))

    # 随机种子。
    seed = int(os.environ.get("ALIGNN_PRETRAIN_SEED", "0"))

    # dataloader worker 数。
    num_workers = int(os.environ.get("ALIGNN_PRETRAIN_NUM_WORKERS", "4"))

    # 训练精度。
    #
    # 如果你的环境支持 AMP，推荐保留 "16-mixed"。
    # 如果你后面怀疑数值稳定性，再改回 32。
    precision = os.environ.get("ALIGNN_PRETRAIN_PRECISION", "16-mixed")

    # =========================
    # 3. 模型结构相关超参数
    # =========================

    # 图 token 最长长度。
    max_graph_len = int(os.environ.get("ALIGNN_PRETRAIN_MAX_GRAPH_LEN", "300"))

    # 3D grid 边长。
    img_size = int(os.environ.get("ALIGNN_PRETRAIN_IMG_SIZE", "30"))

    # 邻居距离高斯展开维度。
    nbr_fea_len = int(os.environ.get("ALIGNN_PRETRAIN_NBR_FEA_LEN", "64"))

    # 隐藏层维度。
    hid_dim = int(os.environ.get("ALIGNN_PRETRAIN_HID_DIM", "768"))

    # Transformer 层数。
    num_layers = int(os.environ.get("ALIGNN_PRETRAIN_NUM_LAYERS", "12"))

    # Transformer 头数。
    num_heads = int(os.environ.get("ALIGNN_PRETRAIN_NUM_HEADS", "12"))

    # MLP 放大倍数。
    mlp_ratio = int(os.environ.get("ALIGNN_PRETRAIN_MLP_RATIO", "4"))

    # ALIGNN block 层数。
    alignn_layers = int(os.environ.get("ALIGNN_PRETRAIN_ALIGNN_LAYERS", "4"))

    # 图编码器 dropout。
    graph_dropout = float(os.environ.get("ALIGNN_PRETRAIN_GRAPH_DROPOUT", "0.1"))

    # MPP masking 比例。
    mpp_ratio = float(os.environ.get("ALIGNN_PRETRAIN_MPP_RATIO", "0.15"))

    # =========================
    # 4. 设备相关配置
    # =========================

    # 使用单卡。
    devices = int(os.environ.get("ALIGNN_PRETRAIN_DEVICES", "1"))

    # 指定 accelerator。
    accelerator = os.environ.get("ALIGNN_PRETRAIN_ACCELERATOR", "gpu")

    # 不显式走分布式。
    strategy = None

    # =========================
    # 5. 预训练任务开关
    # =========================

    # 构造 loss_names。
    loss_names = build_loss_names()

    # ggm 任务需要 false_grid，所以这里随任务自动决定。
    draw_false_grid = loss_names["ggm"] > 0

    # =========================
    # 6. 数据检查
    # =========================

    # 确保数据目录存在。
    if not root_dataset.exists():
        raise FileNotFoundError(f"root_dataset 不存在：{root_dataset}")

    # 检查和当前任务对应的数据文件是否齐全。
    validate_pretrain_dataset(root_dataset, loss_names)

    # 创建日志目录。
    log_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 7. 组装最终配置
    # =========================

    # 先拿项目默认配置。
    config = _config()

    # 用预训练需要的值覆盖默认配置。
    config.update(
        {
            # 实验名。
            "exp_name": exp_name,
            # 数据根目录。
            "root_dataset": str(root_dataset),
            # 预训练没有下游名字。
            "downstream": downstream,
            # 日志目录。
            "log_dir": str(log_dir),
            # 预训练任务组合。
            "loss_names": loss_names,
            # 最大训练轮数。
            "max_epochs": max_epochs,
            # 有效 batch。
            "batch_size": batch_size,
            # 单卡 micro batch。
            "per_gpu_batchsize": per_gpu_batchsize,
            # 学习率。
            "learning_rate": learning_rate,
            # 权重衰减。
            "weight_decay": weight_decay,
            # dropout。
            "drop_rate": drop_rate,
            # 随机种子。
            "seed": seed,
            # DataLoader worker 数。
            "num_workers": num_workers,
            # 训练精度。
            "precision": precision,
            # 图长度上限。
            "max_graph_len": max_graph_len,
            # 3D grid 大小。
            "img_size": img_size,
            # 邻居特征维度。
            "nbr_fea_len": nbr_fea_len,
            # 隐藏维度。
            "hid_dim": hid_dim,
            # Transformer 层数。
            "num_layers": num_layers,
            # 注意力头数。
            "num_heads": num_heads,
            # MLP ratio。
            "mlp_ratio": mlp_ratio,
            # ALIGNN block 数。
            "alignn_layers": alignn_layers,
            # 图模块 dropout。
            "graph_dropout": graph_dropout,
            # MPP mask ratio。
            "mpp_ratio": mpp_ratio,
            # 是否构造 false grid。
            "draw_false_grid": draw_false_grid,
            # 单机单卡。
            "devices": devices,
            # 使用 GPU。
            "accelerator": accelerator,
            # 单节点。
            "num_nodes": 1,
            # 不强制分布式。
            "strategy": strategy,
            # 预训练可从上一阶段 ckpt 初始化，也可以从头开始。
            "load_path": load_path,
            # 不从中断 checkpoint 恢复。
            "resume_from": None,
            # 不是 test only。
            "test_only": False,
            # 预训练不需要回归标签归一化。
            "mean": None,
            # 预训练不需要回归标签归一化。
            "std": None,
        }
    )

    # =========================
    # 8. 启动前打印关键信息
    # =========================

    # 打印主要配置，方便你确认。
    print("==== ALIGNN hMOF pretraining config ====")

    # 逐项打印最关键的字段。
    for key in [
        "exp_name",
        "root_dataset",
        "downstream",
        "log_dir",
        "loss_names",
        "max_epochs",
        "batch_size",
        "per_gpu_batchsize",
        "learning_rate",
        "weight_decay",
        "drop_rate",
        "max_graph_len",
        "img_size",
        "hid_dim",
        "num_layers",
        "alignn_layers",
        "precision",
        "num_workers",
        "load_path",
    ]:
        print(f"{key}: {config.get(key)}")

    # 打印当前 CUDA_VISIBLE_DEVICES，便于你核对是否切到正确显卡。
    print("\nCUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET"))

    # 打印 CUDA 是否可用。
    print("torch.cuda.is_available() =", torch.cuda.is_available())

    # 如果 CUDA 可用，就打印当前可见设备 0 的名字。
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))

    # =========================
    # 9. 正式启动训练
    # =========================

    # 调用项目统一训练入口。
    moftransformer.run(**config)


# 用法示例 1：
# CUDA_VISIBLE_DEVICES=GPU-c8e79920-bd45-2d86-d5d4-1b622bb89872 python /home/yihaoyu/docker/bag/MOFTransformer_upgrade/MOFTransformer/pretrain_alignn_hmof.py

# 用法示例 2：
# CUDA_VISIBLE_DEVICES=GPU-9fc58bf8-5c1d-170d-d506-ff01af38be1a python /home/yihaoyu/docker/bag/MOFTransformer_upgrade/MOFTransformer/pretrain_alignn_hmof.py
