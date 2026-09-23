# MOFTransformer version 2.2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import mean_absolute_error


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def compute_regression(pl_module, batch, normalizer):
    infer = pl_module.infer(batch)

    regression_head_type = getattr(pl_module, "regression_head_type", "linear")
    if regression_head_type in {"fusion", "gated_residual"}:
        logits = pl_module.regression_head(infer)  # [B, n_targets]
    elif regression_head_type in {
        "descriptor_fusion",
        "descriptor_modal_fusion",
        "descriptor_residual",
    }:
        descriptor_key = getattr(pl_module, "regression_descriptor_key", "")
        if not descriptor_key:
            raise ValueError(f"{regression_head_type} requires regression_descriptor_key")
        if descriptor_key not in batch:
            raise KeyError(
                f"{regression_head_type} expected batch['{descriptor_key}'], "
                f"but available keys are: {sorted(batch.keys())}"
            )
        logits = pl_module.regression_head(infer, batch[descriptor_key])  # [B, n_targets]
    else:
        logits = pl_module.regression_head(infer["cls_feats"])  # [B, n_targets]
    labels = torch.FloatTensor(batch["target"]).to(
        logits.device
    )  # [B] or [B, n_targets]

    # normalize encode if config["mean"] and config["std], else pass
    logits = logits.squeeze(-1)
    labels = normalizer.encode(labels)
    loss = F.mse_loss(logits, labels)

    labels = labels.to(torch.float32)
    logits = logits.to(torch.float32)

    ret = {
        "cif_id": infer["cif_id"],
        "cls_feats": infer["cls_feats"],
        "regression_loss": loss,
        "regression_logits": normalizer.decode(logits),
        "regression_labels": normalizer.decode(labels),
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_regression_loss")(ret["regression_loss"])
    mae = getattr(pl_module, f"{phase}_regression_mae")(
        mean_absolute_error(ret["regression_logits"], ret["regression_labels"])
    )

    if pl_module.write_log:
        pl_module.log(f"regression/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"regression/{phase}/mae", mae, sync_dist=True)

    return ret


def compute_classification(pl_module, batch):
    infer = pl_module.infer(batch)

    logits, binary = pl_module.classification_head(
        infer["cls_feats"]
    )  # [B, output_dim]
    labels = torch.LongTensor(batch["target"]).to(logits.device)  # [B]
    assert len(labels.shape) == 1
    if binary:
        logits = logits.squeeze(dim=-1)
        loss = F.binary_cross_entropy_with_logits(input=logits, target=labels.float())
    else:
        loss = F.cross_entropy(logits, labels)

    ret = {
        "cif_id": infer["cif_id"],
        "cls_feats": infer["cls_feats"],
        "classification_loss": loss,
        "classification_logits": logits,
        "classification_labels": labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_classification_loss")(
        ret["classification_loss"]
    )
    acc = getattr(pl_module, f"{phase}_classification_accuracy")(
        ret["classification_logits"], ret["classification_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"classification/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"classification/{phase}/accuracy", acc, sync_dist=True)

    return ret


def compute_mpp(pl_module, batch):
    infer = pl_module.infer(batch, mask_grid=True)

    mpp_logits = pl_module.mpp_head(infer["grid_feats"])  # [B, max_image_len+2, bins]
    mpp_logits = mpp_logits[
        :, :-1, :
    ]  # ignore volume embedding, [B, max_image_len+1, bins]
    mpp_labels = infer["grid_labels"]  # [B, max_image_len+1, C=1]

    mask = mpp_labels != -100.0  # [B, max_image_len, 1]

    # masking
    mpp_logits = mpp_logits[mask.squeeze(-1)]  # [mask, bins]
    mpp_labels = mpp_labels[mask].long()  # [mask]

    mpp_loss = F.cross_entropy(mpp_logits, mpp_labels)

    ret = {
        "mpp_loss": mpp_loss,
        "mpp_logits": mpp_logits,
        "mpp_labels": mpp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpp_loss")(ret["mpp_loss"])
    acc = getattr(pl_module, f"{phase}_mpp_accuracy")(
        ret["mpp_logits"], ret["mpp_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"mpp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"mpp/{phase}/accuracy", acc, sync_dist=True)

    return ret


def compute_mtp(pl_module, batch):
    infer = pl_module.infer(batch)
    mtp_logits = pl_module.mtp_head(infer["cls_feats"])  # [B, hid_dim]
    mtp_labels = torch.LongTensor(batch["mtp"]).to(mtp_logits.device)  # [B]

    mtp_loss = F.cross_entropy(mtp_logits, mtp_labels)  # [B]

    ret = {
        "mtp_loss": mtp_loss,
        "mtp_logits": mtp_logits,
        "mtp_labels": mtp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mtp_loss")(ret["mtp_loss"])
    acc = getattr(pl_module, f"{phase}_mtp_accuracy")(
        ret["mtp_logits"], ret["mtp_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"mtp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"mtp/{phase}/accuracy", acc, sync_dist=True)

    return ret


def compute_vfp(pl_module, batch):
    infer = pl_module.infer(batch)

    vfp_logits = pl_module.vfp_head(infer["cls_feats"]).squeeze(-1)  # [B]
    vfp_labels = torch.FloatTensor(batch["vfp"]).to(vfp_logits.device)

    assert len(vfp_labels.shape) == 1

    vfp_loss = F.mse_loss(vfp_logits, vfp_labels)
    ret = {
        "vfp_loss": vfp_loss,
        "vfp_logits": vfp_logits,
        "vfp_labels": vfp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vfp_loss")(ret["vfp_loss"])
    mae = getattr(pl_module, f"{phase}_vfp_mae")(
        mean_absolute_error(ret["vfp_logits"], ret["vfp_labels"])
    )

    if pl_module.write_log:
        pl_module.log(f"vfp/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"vfp/{phase}/mae", mae, sync_dist=True)

    return ret


def compute_descriptor_regression(pl_module, batch, task, infer=None):
    if infer is None:
        infer = pl_module.infer(batch)

    head = getattr(pl_module, f"{task}_head")
    task_head_type = getattr(
        pl_module,
        f"{task}_head_type",
        getattr(pl_module, "descriptor_head_type", "linear"),
    )
    if task_head_type == "gated_residual" or str(task_head_type).startswith("modal:"):
        logits = head(infer)
    else:
        logits = head(infer["cls_feats"])
    labels = torch.FloatTensor(batch[task]).to(logits.device)

    loss = F.mse_loss(logits, labels)
    ret = {
        f"{task}_loss": loss,
        f"{task}_logits": logits,
        f"{task}_labels": labels,
    }

    trainer = getattr(pl_module, "trainer", None)
    if trainer is not None and getattr(trainer, "testing", False):
        phase = "test"
    elif trainer is not None and getattr(trainer, "validating", False):
        phase = "val"
    else:
        phase = "train" if pl_module.training else "val"
    loss_metric = getattr(pl_module, f"{phase}_{task}_loss")(ret[f"{task}_loss"])
    mae_metric = getattr(pl_module, f"{phase}_{task}_mae") (
        mean_absolute_error(ret[f"{task}_logits"].flatten(), ret[f"{task}_labels"].flatten())
    )

    if pl_module.write_log:
        pl_module.log(f"{task}/{phase}/loss", loss_metric, sync_dist=True)
        pl_module.log(f"{task}/{phase}/mae", mae_metric, sync_dist=True)

    return ret


def compute_symmetry(pl_module, batch, infer=None):
    if infer is None:
        infer = pl_module.infer(batch)

    def masked_mean(feats, masks):
        masks = masks.to(feats.dtype).unsqueeze(-1)
        denom = masks.sum(dim=1).clamp_min(1.0)
        return (feats * masks).sum(dim=1) / denom

    def cosine_distance(a, b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return 1.0 - torch.sum(a * b, dim=-1)

    base_graph = masked_mean(infer["graph_feats"], infer["graph_masks"])
    base_grid = masked_mean(infer["grid_feats"], infer["grid_masks"])

    weights = getattr(
        pl_module,
        "symmetry_feature_weights",
        {"cls": 1.0, "graph": 0.5, "grid": 0.5, "pred": 0.25},
    )
    view_losses = []
    view_cosines = []

    graph_cache = {
        "graph_token_embeds": infer["graph_token_embeds"],
        "graph_token_masks": infer["graph_token_masks"],
        "mo_labels": infer.get("mo_labels", None),
    }

    for sym_grid in pl_module.build_symmetry_views(batch["grid"]):
        sym_infer = pl_module.infer(
            batch,
            grid_override=sym_grid,
            graph_cache=graph_cache,
        )
        sym_graph = masked_mean(sym_infer["graph_feats"], sym_infer["graph_masks"])
        sym_grid_pool = masked_mean(sym_infer["grid_feats"], sym_infer["grid_masks"])

        cls_dist = cosine_distance(infer["cls_feats"], sym_infer["cls_feats"])
        graph_dist = cosine_distance(base_graph, sym_graph)
        grid_dist = cosine_distance(base_grid, sym_grid_pool)

        total = (
            weights["cls"] * cls_dist
            + weights["graph"] * graph_dist
            + weights["grid"] * grid_dist
        )

        pred_penalty = None
        if weights["pred"] > 0:
            pred_terms = []
            if hasattr(pl_module, "chem_head"):
                task_head_type = getattr(pl_module, "chem_head_type", "linear")
                if task_head_type == "gated_residual" or str(task_head_type).startswith("modal:"):
                    base_pred = pl_module.chem_head(infer)
                    sym_pred = pl_module.chem_head(sym_infer)
                else:
                    base_pred = pl_module.chem_head(infer["cls_feats"])
                    sym_pred = pl_module.chem_head(sym_infer["cls_feats"])
                pred_terms.append(F.mse_loss(base_pred, sym_pred, reduction="none").mean(dim=-1))
            if hasattr(pl_module, "zeopp_head"):
                task_head_type = getattr(pl_module, "zeopp_head_type", "linear")
                if task_head_type == "gated_residual" or str(task_head_type).startswith("modal:"):
                    base_pred = pl_module.zeopp_head(infer)
                    sym_pred = pl_module.zeopp_head(sym_infer)
                else:
                    base_pred = pl_module.zeopp_head(infer["cls_feats"])
                    sym_pred = pl_module.zeopp_head(sym_infer["cls_feats"])
                pred_terms.append(F.mse_loss(base_pred, sym_pred, reduction="none").mean(dim=-1))
            if pred_terms:
                pred_penalty = torch.stack(pred_terms, dim=0).mean(dim=0)
                total = total + weights["pred"] * pred_penalty

        view_losses.append(total)
        view_cosines.append(1.0 - cls_dist)

    symmetry_distance = torch.stack(view_losses, dim=0).mean(dim=0)
    symmetry_loss = symmetry_distance.mean()
    cosine_sim = torch.stack(view_cosines, dim=0).mean(dim=0)

    ret = {
        "symmetry_loss": symmetry_loss,
        "symmetry_distance": symmetry_distance.detach(),
        "symmetry_cosine": cosine_sim.detach(),
    }

    trainer = getattr(pl_module, "trainer", None)
    if trainer is not None and getattr(trainer, "testing", False):
        phase = "test"
    elif trainer is not None and getattr(trainer, "validating", False):
        phase = "val"
    else:
        phase = "train" if pl_module.training else "val"

    loss_metric = getattr(pl_module, f"{phase}_symmetry_loss")(ret["symmetry_loss"])
    dist_metric = getattr(pl_module, f"{phase}_symmetry_mae")(symmetry_loss.detach())

    if pl_module.write_log:
        pl_module.log(f"symmetry/{phase}/loss", loss_metric, sync_dist=True)
        pl_module.log(f"symmetry/{phase}/distance", dist_metric, sync_dist=True)
        pl_module.log(
            f"symmetry/{phase}/cosine",
            cosine_sim.mean().detach(),
            sync_dist=True,
        )

    return ret


def compute_ggm(pl_module, batch):
    pos_len = len(batch["grid"]) // 2
    neg_len = len(batch["grid"]) - pos_len
    ggm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )

    ggm_images = []
    for i, (bti, bfi) in enumerate(zip(batch["grid"], batch["false_grid"])):
        if ggm_labels[i] == 1:
            ggm_images.append(bti)
        else:
            ggm_images.append(bfi)

    ggm_images = torch.stack(ggm_images, dim=0)

    batch = {k: v for k, v in batch.items()}
    batch["grid"] = ggm_images

    infer = pl_module.infer(batch)
    ggm_logits = pl_module.ggm_head(infer["cls_feats"])  # cls_feats
    ggm_loss = F.cross_entropy(ggm_logits, ggm_labels.long())

    ret = {
        "ggm_loss": ggm_loss,
        "ggm_logits": ggm_logits,
        "ggm_labels": ggm_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_ggm_loss")(ret["ggm_loss"])
    acc = getattr(pl_module, f"{phase}_ggm_accuracy")(
        ret["ggm_logits"], ret["ggm_labels"]
    )

    if pl_module.write_log:
        pl_module.log(f"ggm/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"ggm/{phase}/accuracy", acc, sync_dist=True)

    return ret


def compute_moc(pl_module, batch):
    if "bbc" in batch.keys():
        task = "bbc"
    else:
        task = "moc"

    infer = pl_module.infer(batch)
    moc_logits = pl_module.moc_head(
        infer["graph_feats"][:, 1:, :]
    ).flatten()  # [B, max_graph_len] -> [B * max_graph_len]
    moc_labels = (
        infer["mo_labels"].to(moc_logits).flatten()
    )  # [B, max_graph_len] -> [B * max_graph_len]
    mask = moc_labels != -100

    moc_loss = F.binary_cross_entropy_with_logits(
        input=moc_logits[mask], target=moc_labels[mask]
    )  # [B * max_graph_len]

    ret = {
        "moc_loss": moc_loss,
        "moc_logits": moc_logits,
        "moc_labels": moc_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_{task}_loss")(ret["moc_loss"])
    acc = getattr(pl_module, f"{phase}_{task}_accuracy")(
        nn.Sigmoid()(ret["moc_logits"]), ret["moc_labels"].long()
    )

    if pl_module.write_log:
        pl_module.log(f"{task}/{phase}/loss", loss, sync_dist=True)
        pl_module.log(f"{task}/{phase}/accuracy", acc, sync_dist=True)

    return ret
