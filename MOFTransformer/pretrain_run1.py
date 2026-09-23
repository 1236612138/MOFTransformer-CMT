# -*- coding: utf-8 -*-
import os
import multiprocessing as mp
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    StochasticWeightAveraging,
)
from sacred.config.config_summary import ConfigSummary

from moftransformer.config import config as _config
from moftransformer.datamodules.datamodule import Datamodule
from moftransformer.modules.module import Module
from moftransformer.utils.validation import get_valid_config

# 允许安全反序列化
torch.serialization.add_safe_globals([ConfigSummary])

# ========== 清理 ckpt ==========
class CleanCheckpointCallback(pl.Callback):
    def __init__(self, keep_top_k=1):
        super().__init__()
        self.keep_top_k = keep_top_k

    def on_validation_end(self, trainer, pl_module):
        ckpt_dirs = []
        for cb in getattr(trainer, "callbacks", []):
            if isinstance(cb, ModelCheckpoint) and cb.dirpath is not None:
                ckpt_dirs.append(Path(cb.dirpath))
        for ckpt_dir in ckpt_dirs:
            if not ckpt_dir.exists():
                continue
            checkpoints = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
            if len(checkpoints) > self.keep_top_k:
                for old in checkpoints[:-self.keep_top_k]:
                    try:
                        old.unlink()
                        print(f"[CleanCheckpoint] Deleted: {old}")
                    except Exception as e:
                        print(f"[CleanCheckpoint] Failed to delete {old}: {e}")

# ========== 子类化：重写优化器 & 修复 lr_scheduler_step ==========
class AggressiveModule(Module):
    def __init__(self, config):
        super().__init__(config)
        self._init_lr = float(config.get("lr", 1e-4))
        self._weight_decay = float(config.get("weight_decay", 5e-2))
        self.save_hyperparameters({"lr": self._init_lr, "weight_decay": self._weight_decay})

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self._init_lr,
            weight_decay=self._weight_decay,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=3,
            threshold=1e-4,
            verbose=True,  # 旧版会 warn，不影响功能
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "regression/val/mae_epoch",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # 关键修复：Lightning 会把 (scheduler, metric) 传进来
    def lr_scheduler_step(self, scheduler, metric):
        # ReduceLROnPlateau 需要 metrics；其他 scheduler 走默认
        if isinstance(scheduler, ReduceLROnPlateau):
            # metric 可能是 tensor，取标量
            if hasattr(metric, "item"):
                metric = metric.item()
            scheduler.step(metric)
        else:
            # 兼容性：有的调度器支持 epoch 参数，这里直接默认 step()
            scheduler.step()

if __name__ == "__main__":
    # ====== 数据与路径 ======
    root_dataset = "/home/yin/bag/MOFTransformer_upgrade/MOFTransformer/all_O2"
    downstream = "O2"
    log_dir = "/home/yin/bag/MOFTransformer_upgrade/MOFTransformer/all_O2/logs_from_scratch"

    # ====== 配置（CPU + 稳定收敛）======
    config = _config()
    cpu_workers = max(2, min(8, mp.cpu_count() // 2))
    config.update({
        "root_dataset": root_dataset,
        "downstream": downstream,
        "log_dir": log_dir,
        "exp_name": "O2_scratch_run_cpu_aggressive_fix",

        "max_epochs": 100,
        "batch_size": 8,
        "per_gpu_batchsize": 8,

        "mean": 0,
        "std": 1,

        "devices": 1,
        "accelerator": "cpu",
        "precision": 32,     # 重要：CPU 上关 AMP，用 float32

        "num_nodes": 1,
        "num_workers": cpu_workers,

        "strategy": "auto",
        "draw_false_grid": False,
        "img_size": 30,
        "nbr_fea_len": 64,
        "seed": 42,

        "val_check_interval": 1.0,

        "load_path": None,
        "resume_from": None,

        # 我们的优化器超参
        "lr": 1e-4,
        "weight_decay": 5e-2,
    })

    # 关键键检查
    for k in ["num_workers", "per_gpu_batchsize", "draw_false_grid", "img_size", "nbr_fea_len"]:
        if k not in config:
            raise KeyError(f"缺少参数: {k}")

    print("✅ 正在从头开始（CPU）训练 MOFTransformer（激进稳定版，已修复 Plateau 调度）...")
    pl.seed_everything(config["seed"])
    config = get_valid_config(config)

    datamodule = Datamodule(config)
    model = AggressiveModule(config)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=log_dir,
        name=config["exp_name"],
        flush_secs=120,
    )

    # 回调
    early_stop = EarlyStopping(
        monitor="regression/val/mae_epoch",
        mode="min",
        patience=8,
        min_delta=1e-4,
        verbose=True,
    )

    # 注意：filename 不再包含带斜杠的指标名，避免奇怪文件名
    checkpoint_callback = ModelCheckpoint(
        monitor="regression/val/mae_epoch",
        mode="min",
        save_top_k=1,
        save_last=False,
        filename="best-epoch={epoch:02d}",
        save_on_train_epoch_end=False,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    swa_cb = StochasticWeightAveraging(swa_lrs=5e-6)
    clean_cb = CleanCheckpointCallback(keep_top_k=1)

    trainer = pl.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        strategy=config["strategy"],
        max_epochs=config["max_epochs"],
        precision=config["precision"],

        gradient_clip_val=1.0,
        deterministic=True,

        log_every_n_steps=10,
        val_check_interval=config["val_check_interval"],
        num_sanity_val_steps=0,
        enable_progress_bar=True,

        callbacks=[checkpoint_callback, early_stop, lr_monitor, swa_cb, clean_cb],
        logger=logger,
    )

    # 训练
    trainer.fit(model, datamodule=datamodule)

    # 测试
    best_path = None
    if hasattr(trainer, "checkpoint_callback"):
        best_path = getattr(trainer.checkpoint_callback, "best_model_path", None)
    if not best_path:
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                best_path = cb.best_model_path or best_path

    if best_path:
        print(f"✅ 正在用最佳模型测试: {best_path}")
        trainer.test(model=None, datamodule=datamodule, ckpt_path=best_path)
    else:
        print("⚠️ 未找到最佳检查点，使用当前权重进行测试")
        trainer.test(model=model, datamodule=datamodule)
