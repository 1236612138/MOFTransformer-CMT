# -*- coding: utf-8 -*-
"""
最终版：加载预训练 ckpt → 两阶段微调（CPU）
要点：
- 两阶段统一使用 nn.Linear 的 regression_head（避免 fc/weight 键名不一致）
- Stage1：冻结 backbone，只训 head
- Stage2：解冻全模型，head 较大学习率 + backbone 小学习率
- ReduceLROnPlateau + EarlyStopping + SWA + 最优模型保存 + 旧 ckpt 清理
- 兼容 ReduceLROnPlateau 的 lr_scheduler_step(metric) 调用
"""

import multiprocessing as mp
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch import nn
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

# 允许 Sacred 对象安全反序列化
torch.serialization.add_safe_globals([ConfigSummary])


# ----------------- Finetune Module（统一 Linear 头 & 兼容 Plateau） -----------------
class FinetuneModule(Module):
    def __init__(
        self,
        config: dict,
        stage: str = "head",             # "head" / "full"
        lr_head: float = 5e-4,           # 头部学习率（两阶段都会用）
        lr_backbone: float = 1e-5,       # 骨干学习率（仅 Stage2）
        weight_decay: float = 1e-2,
        head_out_dim: int = 1,
        force_linear_head: bool = True,  # 两阶段都强制把 head 换成 nn.Linear
    ):
        super().__init__(config)
        self.stage = stage
        self.lr_head = lr_head
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.head_out_dim = head_out_dim
        self.force_linear_head = force_linear_head

        if self.force_linear_head:
            self._ensure_linear_head(self.head_out_dim)

    def _ensure_linear_head(self, out_dim: int = 1):
        """把 regression_head 统一替换为 nn.Linear，确保 state_dict 键为 regression_head.weight/bias"""
        in_features = None
        # 情况1：项目原生 RegressionHead，内部常为 .fc: Linear
        if hasattr(self, "regression_head") and hasattr(self.regression_head, "fc"):
            if isinstance(self.regression_head.fc, nn.Linear):
                in_features = self.regression_head.fc.in_features
        # 情况2：之前已被替换为 Linear
        if in_features is None and hasattr(self, "regression_head") and isinstance(self.regression_head, nn.Linear):
            in_features = self.regression_head.in_features
        # 兜底（按你模型隐藏维度；常见 768，如有不同可改）
        if in_features is None:
            in_features = 768

        self.regression_head = nn.Linear(in_features, out_dim)

    def replace_regression_head(self, out_dim: int = 1):
        """公开接口：必要时再次替换 head（例如切任务时）"""
        self._ensure_linear_head(out_dim)

    def configure_optimizers(self):
        # Stage1：冻结 backbone，只训 head
        if self.stage == "head":
            for name, p in self.named_parameters():
                p.requires_grad = ("regression_head" in name)
            params = [p for p in self.parameters() if p.requires_grad]
            optimizer = AdamW(
                params, lr=self.lr_head, weight_decay=self.weight_decay,
                betas=(0.9, 0.98), eps=1e-8
            )
        else:
            # Stage2：全模型；head 用较大学习率，backbone 用较小学习率
            for p in self.parameters():
                p.requires_grad = True
            head_params, backbone_params = [], []
            for name, p in self.named_parameters():
                (head_params if "regression_head" in name else backbone_params).append(p)
            optimizer = AdamW(
                [
                    {"params": head_params, "lr": self.lr_head},
                    {"params": backbone_params, "lr": self.lr_backbone},
                ],
                weight_decay=self.weight_decay,
                betas=(0.9, 0.98),
                eps=1e-8,
            )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=2,
            threshold=1e-4,
            verbose=True,
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

    # 关键：兼容 ReduceLROnPlateau 的签名
    def lr_scheduler_step(self, scheduler, metric):
        if isinstance(scheduler, ReduceLROnPlateau):
            if hasattr(metric, "item"):
                metric = metric.item()
            scheduler.step(metric)
        else:
            scheduler.step()


# ----------------- 仅保留最新 ckpt 的清理回调 -----------------
class CleanCheckpointCallback(pl.Callback):
    def __init__(self, keep_top_k=1):
        super().__init__()
        self.keep_top_k = keep_top_k

    def on_validation_end(self, trainer, pl_module):
        from pytorch_lightning.callbacks import ModelCheckpoint
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.dirpath:
                ckpt_dir = Path(cb.dirpath)
                if not ckpt_dir.exists():
                    continue
                checkpoints = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
                for old in checkpoints[:-self.keep_top_k]:
                    try:
                        old.unlink()
                        print(f"[CleanCheckpoint] Deleted: {old}")
                    except Exception as e:
                        print(f"[CleanCheckpoint] Failed to delete {old}: {e}")


if __name__ == "__main__":
    # ----------------- 路径配置 -----------------
    root_dataset = "/home/yin/bag/MOFTransformer_upgrade/MOFTransformer/all_O2"
    downstream = "O2"  # 微调任务
    log_root = "/home/yin/bag/MOFTransformer_upgrade/MOFTransformer"

    # ★ 把这个路径改成你的“预训练最佳 ckpt”
    pretrained_ckpt = "/home/yin/bag/MOFTransformer_upgrade/MOFTransformer/all_O2/logs_from_scratch/O2_scratch_run_cpu_aggressive_fix/version_0/checkpoints/best-epoch=epoch=38.ckpt"
    assert Path(pretrained_ckpt).exists(), f"找不到预训练 ckpt: {pretrained_ckpt}"

    # ----------------- 通用 config（CPU 稳定） -----------------
    config = _config()
    cpu_workers = max(2, min(8, mp.cpu_count() // 2))
    config.update({
        "root_dataset": root_dataset,
        "downstream": downstream,
        "log_dir": f"{log_root}/finetune_logs",
        "exp_name": "O2_finetune_from_pretrained_cpu",

        "max_epochs": 100,          # 由 EarlyStopping 控制实际轮数
        "batch_size": 2,
        "per_gpu_batchsize": 2,     # 兼容项目内部逻辑

        "mean": 0,
        "std": 1,

        "devices": 1,
        "accelerator": "cpu",
        "precision": 32,            # CPU 上 float32 最稳
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
    })

    # 关键键检查（与你项目保持一致）
    for k in ["num_workers", "per_gpu_batchsize", "draw_false_grid", "img_size", "nbr_fea_len"]:
        if k not in config:
            raise KeyError(f"缺少参数: {k}")

    # 固定随机性
    pl.seed_everything(config["seed"])

    # Datamodule 用“字典对象”构造（不要 **kwargs）
    valid_config = get_valid_config(config)
    datamodule = Datamodule(valid_config)

    # ----------------- Stage 1：只训练 Linear head -----------------
    model_stage1 = FinetuneModule.load_from_checkpoint(
        checkpoint_path=pretrained_ckpt,   # 加载预训练权重
        config=valid_config,
        stage="head",
        lr_head=5e-4,
        lr_backbone=1e-5,                  # Stage1 用不到
        weight_decay=1e-2,
        head_out_dim=1,
        force_linear_head=True,            # 构造时即 Linear 头
        strict=False,                      # 兜底少量键差异
    )
    # 切换任务时，明确重置 head（确保不沿用预训练 head 权重）
    model_stage1.replace_regression_head(out_dim=1)

    logger1 = pl.loggers.TensorBoardLogger(
        save_dir=valid_config["log_dir"],
        name=valid_config["exp_name"] + "_stage1_head",
        flush_secs=120,
    )
    ckpt1 = ModelCheckpoint(
        monitor="regression/val/mae_epoch",
        mode="min",
        filename="best-epoch={epoch:02d}",
        save_top_k=1,
        save_last=False,
        verbose=True,
        save_on_train_epoch_end=False,
    )
    early1 = EarlyStopping(monitor="regression/val/mae_epoch", mode="min", patience=8, min_delta=1e-4, verbose=True)
    lrmon1 = LearningRateMonitor(logging_interval="epoch")
    swa1 = StochasticWeightAveraging(swa_lrs=1e-5)
    clean1 = CleanCheckpointCallback(keep_top_k=1)

    trainer1 = pl.Trainer(
        accelerator="cpu",
        devices=1,
        precision=32,
        max_epochs=40,                 # 阶段1上限
        deterministic=True,
        log_every_n_steps=10,
        val_check_interval=1.0,
        num_sanity_val_steps=0,
        callbacks=[ckpt1, early1, lrmon1, swa1, clean1],
        logger=logger1,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
    )

    print("=== [Stage 1] 只训练新 Linear head ===")
    trainer1.fit(model_stage1, datamodule=datamodule)
    stage1_best = ckpt1.best_model_path

    # ----------------- Stage 2：解冻全模型（同样 Linear 头结构） -----------------
    if stage1_best:
        model_stage2 = FinetuneModule.load_from_checkpoint(
            checkpoint_path=stage1_best,
            config=valid_config,
            stage="full",
            lr_head=1e-4,            # head 小一点继续收敛
            lr_backbone=1e-5,        # backbone 更小
            weight_decay=1e-2,
            head_out_dim=1,
            force_linear_head=True,  # 保持 Linear 结构与 ckpt 一致
            strict=False,
        )
    else:
        print("⚠️ 阶段1无最优 ckpt，使用当前权重进入阶段2")
        model_stage2 = FinetuneModule(
            valid_config,
            stage="full",
            lr_head=1e-4,
            lr_backbone=1e-5,
            weight_decay=1e-2,
            head_out_dim=1,
            force_linear_head=True,
        )

    logger2 = pl.loggers.TensorBoardLogger(
        save_dir=valid_config["log_dir"],
        name=valid_config["exp_name"] + "_stage2_full",
        flush_secs=120,
    )
    ckpt2 = ModelCheckpoint(
        monitor="regression/val/mae_epoch",
        mode="min",
        filename="best-epoch={epoch:02d}",
        save_top_k=1,
        save_last=False,
        verbose=True,
        save_on_train_epoch_end=False,
    )
    early2 = EarlyStopping(monitor="regression/val/mae_epoch", mode="min", patience=10, min_delta=1e-4, verbose=True)
    lrmon2 = LearningRateMonitor(logging_interval="epoch")
    swa2 = StochasticWeightAveraging(swa_lrs=5e-6)
    clean2 = CleanCheckpointCallback(keep_top_k=1)

    trainer2 = pl.Trainer(
        accelerator="cpu",
        devices=1,
        precision=32,
        max_epochs=60,                 # 阶段2上限
        deterministic=True,
        log_every_n_steps=10,
        val_check_interval=1.0,
        num_sanity_val_steps=0,
        callbacks=[ckpt2, early2, lrmon2, swa2, clean2],
        logger=logger2,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
    )

    print("=== [Stage 2] 解冻全模型，小 LR 微调（Linear head） ===")
    trainer2.fit(model_stage2, datamodule=datamodule)

    # ----------------- 测试（使用最终最优 ckpt） -----------------
    final_ckpt = ckpt2.best_model_path or stage1_best
    if final_ckpt:
        print(f"✅ 用最优 ckpt 测试：{final_ckpt}")
        trainer2.test(model=None, datamodule=datamodule, ckpt_path=final_ckpt)
    else:
        print("⚠️ 无最优 ckpt，使用当前权重测试")
        trainer2.test(model=model_stage2, datamodule=datamodule)
