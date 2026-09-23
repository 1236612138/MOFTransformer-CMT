import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt
from sacred.config.config_summary import ConfigSummary

import moftransformer
from moftransformer.config import config as _config
from moftransformer.datamodules.datamodule import Datamodule
from moftransformer.modules.module import Module
from moftransformer.utils.validation import get_valid_config

# 注册 Safe Unpickling 的类
torch.serialization.add_safe_globals([ConfigSummary])

if __name__ == "__main__":
    # ====== 数据与路径设置 ======
    root_dataset = "/home/yin/bag/MOFTransformer_upgrade/MOFTransformer/all_O2"
    downstream = "O2"
    log_dir = "/home/yin/bag/MOFTransformer_upgrade/MOFTransformer/all_O2/logs_from_scratch"
    
    # ====== 超参数设置 ======
    config = _config()
    config.update({
        "root_dataset": root_dataset,
        "downstream": downstream,
        "log_dir": log_dir,
        "exp_name": "O2_scratch_run",   # 实验名
        "max_epochs": 50,
        "batch_size": 8,
        "per_gpu_batchsize": 8,
        "mean": 0,
        "std": 1,
        "devices": 1,
        "accelerator": "cpu",            # ⚠️ 如果你有 GPU，换成 "gpu"
        "num_nodes": 1,
        "precision": 32,
        "num_workers": 0,
        "strategy": "auto",              # Lightning 2.0 推荐
        "draw_false_grid": False,
        "img_size": 30,
        "nbr_fea_len": 64,
        "seed": 42,
        "val_check_interval": 1.0,

        # 不加载任何预训练权重（从头训练）
        "load_path": None,
        "resume_from": None,
    })

    # ✅ 验证关键参数
    required_keys = ["num_workers", "per_gpu_batchsize", "draw_false_grid", "img_size", "nbr_fea_len"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"缺少参数: {key}")

    # ====== 开始训练 ======
    print("✅ 正在从头开始训练 MOFTransformer ...")
    pl.seed_everything(config["seed"])
    config = get_valid_config(config)
    datamodule = Datamodule(config)
    model = Module(config)

    # TensorBoard 日志保存
    logger = pl.loggers.TensorBoardLogger(
        save_dir=log_dir,
        name=config["exp_name"]
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/the_metric",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        strategy=config["strategy"],
        max_epochs=config["max_epochs"],
        precision=config["precision"],
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=config["val_check_interval"],
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)

    # ====== 测试最佳模型 ======
    best_ckpt = Path(logger.log_dir) / "checkpoints" / "best.ckpt"
    print(f"✅ 正在用 best.ckpt 测试: {best_ckpt}")
    trainer.test(model, datamodule=datamodule, ckpt_path=str(best_ckpt))
