# MOFTransformer version 2.2.0
import torch

from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)

from moftransformer.gadgets.my_metrics import Accuracy, Scalar


def set_metrics(pl_module):
    """
    Create metric containers for train/val/test.

    NOTE:
    Some older objectives may still update metrics only based on `pl_module.training`
    (train vs val). To keep backward compatibility, epoch_wrapup() will prefer test_*
    but fallback to val_* if test_* has no updates.
    """
    for split in ["train", "val", "test"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v <= 0:
                continue
            if k in ("regression", "vfp", "chem", "zeopp", "topo", "symmetry"):
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_mae", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v > 0
    ]
    return


def _get_stage(pl_module) -> str:
    """
    Determine stage: train / val / test.
    Prefer trainer flags when available.
    """
    tr = getattr(pl_module, "trainer", None)
    if tr is not None:
        if getattr(tr, "testing", False):
            return "test"
        if getattr(tr, "validating", False):
            return "val"
        if getattr(tr, "training", False):
            return "train"

    # fallback to old logic
    return "train" if pl_module.training else "val"


def _metric_has_updates(metric_obj) -> bool:
    """
    Scalar and Accuracy both have `total` state in your implementation.
    - Scalar: total counts number of updates
    - Accuracy: total counts number of valid labels
    """
    try:
        return metric_obj.total.detach().item() > 0
    except Exception:
        return False


def _pick_split(pl_module, stage: str, loss_name: str, kind: str) -> str:
    """
    Choose which split metric to read/reset from.

    If stage is test:
      prefer test_* if it has updates, else fallback to val_* (compat with old objectives).
    """
    if stage in ("train", "val"):
        return stage

    # stage == "test"
    test_attr = f"test_{loss_name}_{kind}"
    val_attr = f"val_{loss_name}_{kind}"

    test_obj = getattr(pl_module, test_attr, None)
    if test_obj is not None and _metric_has_updates(test_obj):
        return "test"

    if getattr(pl_module, val_attr, None) is not None:
        return "val"

    return "test"


def epoch_wrapup(pl_module):
    """
    Log epoch-level metrics for the current stage.

    Outputs:
      - train: {loss_name}/train/* and train/the_metric
      - val:   {loss_name}/val/*   and val/the_metric
      - test:  {loss_name}/test/*  and test/the_metric   (FIX)
    """
    stage = _get_stage(pl_module)  # train / val / test
    the_metric = 0

    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v <= 0:
            continue

        if loss_name in ("regression", "vfp", "chem", "zeopp", "topo", "symmetry"):
            # ---- loss_epoch ----
            split_for_loss = _pick_split(pl_module, stage, loss_name, "loss")
            loss_obj = getattr(pl_module, f"{split_for_loss}_{loss_name}_loss")
            pl_module.log(
                f"{loss_name}/{stage}/loss_epoch",
                loss_obj.compute(),
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            loss_obj.reset()

            # ---- mae_epoch ----
            split_for_mae = _pick_split(pl_module, stage, loss_name, "mae")
            mae_obj = getattr(pl_module, f"{split_for_mae}_{loss_name}_mae")
            value = mae_obj.compute()
            pl_module.log(
                f"{loss_name}/{stage}/mae_epoch",
                value,
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            mae_obj.reset()

            # the_metric uses -MAE (so ModelCheckpoint mode="max" selects smaller MAE)
            value = -value

        else:
            # ---- accuracy_epoch ----
            split_for_acc = _pick_split(pl_module, stage, loss_name, "accuracy")
            acc_obj = getattr(pl_module, f"{split_for_acc}_{loss_name}_accuracy")
            value = acc_obj.compute()
            pl_module.log(
                f"{loss_name}/{stage}/accuracy_epoch",
                value,
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            acc_obj.reset()

            # ---- loss_epoch ----
            split_for_loss = _pick_split(pl_module, stage, loss_name, "loss")
            loss_obj = getattr(pl_module, f"{split_for_loss}_{loss_name}_loss")
            pl_module.log(
                f"{loss_name}/{stage}/loss_epoch",
                loss_obj.compute(),
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            loss_obj.reset()

        the_metric += value

    pl_module.log(f"{stage}/the_metric", the_metric, sync_dist=True)


def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]
    backbone_wd = pl_module.hparams.config.get("backbone_weight_decay", wd)
    graph_wd = pl_module.hparams.config.get("graph_weight_decay", wd)
    head_wd = pl_module.hparams.config.get("head_weight_decay", wd)

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["regression_head", "classification_head", "chem_head", "zeopp_head"]
    graph_names = [
        "graph_embeddings",
        "token_type_embeddings",
        "cls_embeddings",
        "volume_embeddings",
        "pooler",
    ]
    head_lr_mult = pl_module.hparams.config.get("head_lr_mult", pl_module.hparams.config["lr_mult"])
    graph_lr_mult = pl_module.hparams.config.get("graph_lr_mult", 1.0)
    backbone_lr_mult = pl_module.hparams.config.get("backbone_lr_mult", 1.0)
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]

    def _is_no_decay(name):
        return any(nd in name for nd in no_decay)

    def _is_head(name):
        return any(hn in name for hn in head_names)

    def _is_graph(name):
        return any(gn in name for gn in graph_names)

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if p.requires_grad
                and not _is_no_decay(n)
                and not _is_head(n)
                and not _is_graph(n)
            ],
            "weight_decay": backbone_wd,
            "lr": lr * backbone_lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if p.requires_grad
                and _is_no_decay(n)
                and not _is_head(n)
                and not _is_graph(n)
            ],
            "weight_decay": 0.0,
            "lr": lr * backbone_lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if p.requires_grad
                and not _is_no_decay(n)
                and _is_graph(n)
                and not _is_head(n)
            ],
            "weight_decay": graph_wd,
            "lr": lr * graph_lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if p.requires_grad
                and _is_no_decay(n)
                and _is_graph(n)
                and not _is_head(n)
            ],
            "weight_decay": 0.0,
            "lr": lr * graph_lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if p.requires_grad
                and not _is_no_decay(n)
                and _is_head(n)
            ],
            "weight_decay": head_wd,
            "lr": lr * head_lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if p.requires_grad and _is_no_decay(n) and _is_head(n)
            ],
            "weight_decay": 0.0,
            "lr": lr * head_lr_mult,
        },
    ]
    optimizer_grouped_parameters = [
        group for group in optimizer_grouped_parameters if group["params"]
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optim_type: {optim_type}")

    if pl_module.trainer.max_steps == -1:
        max_steps = pl_module.trainer.estimated_stepping_batches
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(warmup_steps, float):
        warmup_steps = int(max_steps * warmup_steps)

    print(
        f"max_epochs: {pl_module.trainer.max_epochs} | max_steps: {max_steps} | warmup_steps : {warmup_steps} "
        f"| weight_decay(backbone/graph/head): {backbone_wd}/{graph_wd}/{head_wd} "
        f"| lr_mult(backbone/graph/head): {backbone_lr_mult}/{graph_lr_mult}/{head_lr_mult} "
        f"| decay_power : {decay_power}"
    )

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    elif decay_power == "constant":
        scheduler = get_constant_schedule(optimizer)
    elif decay_power == "constant_with_warmup":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}
    return [optimizer], [sched]


class Normalizer(object):
    """
    normalize for regression
    """

    def __init__(self, mean, std, device):
        # FIX: mean=0 should still enable normalization (0,1)
        if (mean is not None) and (std is not None):
            if isinstance(mean, list):
                mean = torch.tensor(mean).to(device)
            if isinstance(std, list):
                std = torch.tensor(std).to(device)
            self.mean = mean
            self.std = std
            self._norm_func = lambda tensor: (tensor - mean) / std
            self._denorm_func = lambda tensor: tensor * std + mean
        else:
            self._norm_func = lambda tensor: tensor
            self._denorm_func = lambda tensor: tensor

    def encode(self, tensor):
        return self._norm_func(tensor)

    def decode(self, tensor):
        return self._denorm_func(tensor)
