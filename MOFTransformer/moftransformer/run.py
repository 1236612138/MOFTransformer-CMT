# MOFTransformer version 2.2.0
import sys
import os
import copy
import warnings
from pathlib import Path
import shutil

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging

from moftransformer.config import ex
from moftransformer.config import config as _config
from moftransformer.datamodules.datamodule import Datamodule
from moftransformer.modules.module import Module, _load_ckpt_with_alignn_compat
from moftransformer.utils.validation import (
    get_valid_config,
    get_num_devices,
)

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

_IS_INTERACTIVE = hasattr(sys, "ps1")


def run(root_dataset, downstream=None, log_dir="logs/", *, test_only=False, **kwargs):
    """
    Train or predict MOFTransformer.
    """

    # 1) base config
    config = copy.deepcopy(_config())

    # 2) merge kwargs (try.py 传进来的覆盖默认)
    for key in kwargs.keys():
        if key not in config:
            print(f"Warning: {key} is not in the default configuration.")
        config[key] = kwargs[key]

    # 3) required fields
    config["root_dataset"] = root_dataset
    config["downstream"] = downstream
    config["log_dir"] = log_dir
    config["test_only"] = test_only

    # ------------------------------------------------------------
    # ✅ 关键修复：不要再用 gpus=0 去强行把 accelerator 改成 cpu
    # 兼容旧参数：如果用户传了 gpus 且没显式传 devices/accelerator，则用 gpus 推导
    # ------------------------------------------------------------
    user_set_acc = "accelerator" in kwargs
    user_set_dev = "devices" in kwargs

    if (not user_set_acc) and (not user_set_dev):
        # 兼容旧逻辑：优先读 gpus
        g = config.get("gpus", None)
        if isinstance(g, int) and g > 0:
            config["accelerator"] = "gpu"
            config["devices"] = g
        else:
            # 没给 gpus，也没显式给 accelerator/devices，则按环境自动
            config.setdefault("accelerator", "auto")
            config.setdefault("devices", 1)

    # 默认不要分布式
    config.setdefault("distributed_backend", "none")

    # 如果用户要用 gpu，但机器不可用，就直接报清楚
    if str(config.get("accelerator", "")).lower() in ["gpu", "cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "You set accelerator='gpu' but torch.cuda.is_available() is False. "
                "Please check your PyTorch CUDA installation / driver / CUDA_VISIBLE_DEVICES."
            )

    print("Final config:", {k: config.get(k) for k in [
        "root_dataset", "downstream", "log_dir", "test_only",
        "accelerator", "devices", "num_nodes", "strategy", "precision",
        "batch_size", "per_gpu_batchsize", "num_workers"
    ]})

    main(config)


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    _config = get_valid_config(_config)
    dm = Datamodule(_config)
    model = Module(_config)
    exp_name = f"{_config['exp_name']}"

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )

    if _config["test_only"]:
        name = f'test_{exp_name}_seed{_config["seed"]}_from_{str(_config["load_path"]).split("/")[-1][:-5]}'
    else:
        name = f'{exp_name}_seed{_config["seed"]}_from_{str(_config["load_path"]).split("/")[-1][:-5]}'

    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=name,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    early_stop_patience = _config.get("early_stop_patience")
    early_stop_min_delta = _config.get("early_stop_min_delta", 1e-4)
    if early_stop_patience is not None and int(early_stop_patience) > 0:
        callbacks.append(
            EarlyStopping(
                monitor="regression/val/mae_epoch"
                if _config["loss_names"].get("regression", 0) > 0
                else "val/the_metric",
                mode="min"
                if _config["loss_names"].get("regression", 0) > 0
                else "max",
                patience=int(early_stop_patience),
                min_delta=float(early_stop_min_delta),
                verbose=True,
            )
        )

    swa_lrs = _config.get("swa_lrs")
    if swa_lrs is not None:
        callbacks.append(StochasticWeightAveraging(swa_lrs=float(swa_lrs)))

    num_device = get_num_devices(_config)
    print("num_device", num_device)

    # gradient accumulation
    if num_device == 0:
        accumulate_grad_batches = _config["batch_size"] // (
            _config["per_gpu_batchsize"] * _config["num_nodes"]
        )
    else:
        accumulate_grad_batches = _config["batch_size"] // (
            _config["per_gpu_batchsize"] * num_device * _config["num_nodes"]
        )

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    log_every_n_steps = 10
    num_sanity_val_steps = int(_config.get("num_sanity_val_steps", 2))
    enable_progress_bar = _config.get("enable_progress_bar")
    if enable_progress_bar is None:
        env_progress = os.environ.get("MOF_ENABLE_PROGRESS_BAR", "").strip().lower()
        if env_progress in {"1", "true", "yes", "on"}:
            enable_progress_bar = True
        elif env_progress in {"0", "false", "no", "off"}:
            enable_progress_bar = False
        else:
            enable_progress_bar = sys.stdout.isatty()

    trainer_kwargs = dict(
        accelerator=_config.get("accelerator", "auto"),
        devices=_config.get("devices", 1),
        num_nodes=_config.get("num_nodes", 1),
        precision=_config.get("precision", 32),
        benchmark=True,
        max_epochs=_config["max_epochs"],
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=_config["val_check_interval"],
        num_sanity_val_steps=num_sanity_val_steps,
        deterministic=True,
        enable_progress_bar=enable_progress_bar,
        gradient_clip_val=float(_config.get("gradient_clip_val", 0.0)),
    )

    # ✅ strategy：只有你显式给了才传（避免 Lightning 自己起 DDP）
    if isinstance(_config.get("strategy"), str) and _config["strategy"] not in ["", "auto", "ddp", "ddp_spawn"]:
        trainer_kwargs["strategy"] = _config["strategy"]

    trainer = pl.Trainer(**trainer_kwargs)

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm, ckpt_path=_config["resume_from"])

        best_path = None
        if hasattr(trainer, "checkpoint_callback"):
            best_path = getattr(trainer.checkpoint_callback, "best_model_path", None)
        if not best_path:
            for cb in trainer.callbacks:
                if isinstance(cb, pl.callbacks.ModelCheckpoint):
                    best_path = cb.best_model_path or best_path

        # 用“只恢复模型权重、不恢复 callback 状态”的方式测试，
        # 避免 SWA callback 在恢复 average_model_state 时因为键不匹配而报错。
        if best_path:
            print(f"Testing with best checkpoint weights only: {best_path}")
            test_model = Module(_config)
            _load_ckpt_with_alignn_compat(test_model, best_path, test_only=True)
            trainer.test(test_model, datamodule=dm, ckpt_path=None)
        else:
            print("Warning: best checkpoint not found, testing current model weights.")
            trainer.test(model, datamodule=dm, ckpt_path=None)

        log_dir = Path(logger.log_dir) / "checkpoints"
        if best_path and Path(best_path).exists():
            shutil.copy(best_path, log_dir / "best.ckpt")
    else:
        trainer.test(model, datamodule=dm)
