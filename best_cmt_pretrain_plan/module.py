# MOFTransformer version 2.1.0
from typing import Any, List
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from moftransformer.modules import objectives, heads, module_utils
from moftransformer.modules.alignn import ALIGNNGraphEmbeddings as GraphEmbeddings

from moftransformer.modules.vision_transformer_3d import VisionTransformer3D
from moftransformer.modules.module_utils import Normalizer

import numpy as np
from sklearn.metrics import r2_score


def _load_ckpt_with_alignn_compat(module, load_path: str, test_only: bool = False):
    """
    优先尝试完整加载 ckpt。
    如果遇到旧的 CGCNN/PMTransformer ckpt 与当前 ALIGNN 图编码器不兼容，
    再回退到“跳过 graph_embeddings.*”的兼容模式。
    """
    ckpt = torch.load(load_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    try:
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        mode = "full"
        skipped = []
    except RuntimeError as e:
        skipped_prefixes = ("graph_embeddings.",)
        filtered = {}
        skipped = []
        for k, v in state_dict.items():
            if k.startswith(skipped_prefixes):
                skipped.append(k)
                continue
            filtered[k] = v
        missing, unexpected = module.load_state_dict(filtered, strict=False)
        mode = "skip_graph_embeddings"
        print(f"[ckpt load][ALIGNN] fallback because of RuntimeError: {e}")

    suffix = "[test_only]" if test_only else ""
    print(f"load model : {load_path}")
    print(
        f"[ckpt load][ALIGNN]{suffix} mode={mode} "
        f"missing_keys={len(missing)} unexpected_keys={len(unexpected)} skipped={len(skipped)}"
    )
    if len(skipped) > 0:
        print("  - example skipped:", skipped[:30])
    if len(missing) > 0:
        print("  - example missing:", missing[:30])
    if len(unexpected) > 0:
        print("  - example unexpected:", unexpected[:30])


class Module(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.max_grid_len = config["max_grid_len"]
        self.vis = config["visualize"]

        # graph embedding with_unique_atoms - 使用 ALIGNN 全图嵌入
        self.graph_embeddings = GraphEmbeddings(
            in_feats=119,
            max_graph_len=config["max_graph_len"],
            out_dim=config["hid_dim"],
            num_layers=config.get("alignn_layers", 4),
            dropout=config.get("graph_dropout", 0.1),
            vis=config["visualize"],
        )
        # 这里不再调用 init_weights，让 PyTorch 默认初始化即可
        # 如果你想统一，可以在 objectives.init_weights 里加对 ALIGNN 的处理

        # token type embeddings
        self.token_type_embeddings = nn.Embedding(2, config["hid_dim"])
        self.token_type_embeddings.apply(objectives.init_weights)

        # set transformer
        self.transformer = VisionTransformer3D(
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            in_chans=config["in_chans"],
            embed_dim=config["hid_dim"],
            depth=config["num_layers"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            drop_rate=config["drop_rate"],
            mpp_ratio=config["mpp_ratio"],
        )

        # class token
        self.cls_embeddings = nn.Linear(1, config["hid_dim"])
        self.cls_embeddings.apply(objectives.init_weights)

        # volume token
        self.volume_embeddings = nn.Linear(1, config["hid_dim"])
        self.volume_embeddings.apply(objectives.init_weights)

        # pooler
        self.pooler = heads.Pooler(config["hid_dim"])
        self.pooler.apply(objectives.init_weights)

        # ===================== loss =====================
        if config["loss_names"]["ggm"] > 0:
            self.ggm_head = heads.GGMHead(config["hid_dim"])
            self.ggm_head.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_head = heads.MPPHead(config["hid_dim"])
            self.mpp_head.apply(objectives.init_weights)

        if config["loss_names"]["mtp"] > 0:
            self.mtp_head = heads.MTPHead(config["hid_dim"])
            self.mtp_head.apply(objectives.init_weights)

        if config["loss_names"]["vfp"] > 0:
            self.vfp_head = heads.VFPHead(config["hid_dim"])
            self.vfp_head.apply(objectives.init_weights)

        if config["loss_names"]["moc"] > 0 or config["loss_names"]["bbc"] > 0:
            self.moc_head = heads.MOCHead(config["hid_dim"])
            self.moc_head.apply(objectives.init_weights)

        # ===================== Downstream =====================
        hid_dim = config["hid_dim"]

        # ---------------------------------------------------------------------
        # BEGIN PATCH (可回滚)：ckpt 加载改为“跳过 graph_embeddings.* + 打印 missing/unexpected”
        # 目的：pmtransformer.ckpt 的 graph_embeddings 是 CGCNN 的，和 ALIGNN 不匹配。
        # 原始写法 strict=False 会静默跳过大量权重，你看不到到底加载了什么，导致对比不公平/不可解释。
        # ---------------------------------------------------------------------
        if config["load_path"] != "" and not config["test_only"]:
            _load_ckpt_with_alignn_compat(self, config["load_path"], test_only=False)
        # ---------------------------------------------------------------------
        # END PATCH
        # ---------------------------------------------------------------------

        if self.hparams.config["loss_names"]["regression"] > 0:
            self.regression_head = heads.RegressionHead(hid_dim, config["n_targets"])
            self.regression_head.apply(objectives.init_weights)
            # normalization
            self.mean = config["mean"]
            self.std = config["std"]

        if self.hparams.config["loss_names"]["classification"] > 0:
            n_classes = config["n_classes"]
            self.classification_head = heads.ClassificationHead(hid_dim, n_classes)
            self.classification_head.apply(objectives.init_weights)

        module_utils.set_metrics(self)
        self.current_tasks = list()
        # ===================== load downstream (test_only) ======================

        # ---------------------------------------------------------------------
        # BEGIN PATCH (可回滚)：test_only 情况也保持同样加载规则
        # ---------------------------------------------------------------------
        if config["load_path"] != "" and config["test_only"]:
            _load_ckpt_with_alignn_compat(self, config["load_path"], test_only=True)
        # ---------------------------------------------------------------------
        # END PATCH
        # ---------------------------------------------------------------------

        self.test_logits = []
        self.test_labels = []
        self.test_cifid = []
        self.write_log = True
        self._backbone_frozen = False

    def _freeze_backbone(self):
        head_prefixes = ("regression_head", "classification_head")
        for name, param in self.named_parameters():
            if any(name.startswith(prefix) for prefix in head_prefixes):
                param.requires_grad = True
            else:
                param.requires_grad = False
        self._backbone_frozen = True
        print("[finetune] backbone frozen; only task head parameters are trainable")

    def _unfreeze_backbone(self):
        for _, param in self.named_parameters():
            param.requires_grad = True
        self._backbone_frozen = False
        print("[finetune] backbone unfrozen; full model finetuning enabled")

    def infer(
        self,
        batch,
        mask_grid=False,
    ):
        cif_id = batch["cif_id"]
        atom_num = batch["atom_num"]  # [N']
        nbr_idx = batch["nbr_idx"]  # [N', M]
        nbr_fea = batch["nbr_fea"]  # [N', M, nbr_fea_len]
        crystal_atom_idx = batch["crystal_atom_idx"]  # list [B]
        uni_idx = batch["uni_idx"]  # list [B]
        uni_count = batch["uni_count"]  # list [B]

        angle_src = batch.get("angle_src", None)
        angle_dst = batch.get("angle_dst", None)
        angle_fea = batch.get("angle_fea", None)

        grid = batch["grid"]  # [B, C, H, W, D]
        volume = batch["volume"]  # list [B]

        if "moc" in batch.keys():
            moc = batch["moc"]  # [B]
        elif "bbc" in batch.keys():
            moc = batch["bbc"]  # [B]
        else:
            moc = None

        # get graph embeds (ALIGNN)
        (
            graph_embeds,  # [B, max_graph_len, hid_dim],
            graph_masks,  # [B, max_graph_len],
            mo_labels,  # if moc: [B, max_graph_len], else: None
        ) = self.graph_embeddings(
            atom_num=atom_num,
            nbr_idx=nbr_idx,
            nbr_fea=nbr_fea,
            crystal_atom_idx=crystal_atom_idx,
            uni_idx=uni_idx,
            uni_count=uni_count,
            moc=moc,
            angle_src=angle_src,
            angle_dst=angle_dst,
            angle_fea=angle_fea,
        )

        # add class embeds to graph_embeds
        cls_tokens = torch.zeros(len(crystal_atom_idx), device=graph_embeds.device)  # [B]
        cls_embeds = self.cls_embeddings(cls_tokens[:, None, None])  # [B, 1, hid_dim]
        cls_mask = torch.ones(len(crystal_atom_idx), 1, device=graph_masks.device)  # [B, 1]

        graph_embeds = torch.cat([cls_embeds, graph_embeds], dim=1)  # [B, max_graph_len+1, hid_dim]
        graph_masks = torch.cat([cls_mask, graph_masks], dim=1)  # [B, max_graph_len+1]

        # get grid embeds
        (
            grid_embeds,  # [B, max_grid_len+1, hid_dim]
            grid_masks,  # [B, max_grid_len+1]
            grid_labels,  # [B, grid+1, C] if mask_image == True
        ) = self.transformer.visual_embed(
            grid,
            max_image_len=self.max_grid_len,
            mask_it=mask_grid,
        )

        # add volume embeds to grid_embeds
        volume = torch.FloatTensor(volume).to(grid_embeds.device)  # [B]
        volume_embeds = self.volume_embeddings(volume[:, None, None])  # [B, 1, hid_dim]
        volume_mask = torch.ones(volume.shape[0], 1, device=grid_masks.device)

        grid_embeds = torch.cat([grid_embeds, volume_embeds], dim=1)  # [B, max_grid_len+2, hid_dim]
        grid_masks = torch.cat([grid_masks, volume_mask], dim=1)  # [B, max_grid_len+2]

        # add token_type_embeddings
        graph_embeds = graph_embeds + self.token_type_embeddings(
            torch.zeros_like(graph_masks, device=self.device).long()
        )
        grid_embeds = grid_embeds + self.token_type_embeddings(
            torch.ones_like(grid_masks, device=self.device).long()
        )

        co_embeds = torch.cat([graph_embeds, grid_embeds], dim=1)  # [B, final_max_len, hid_dim]
        co_masks = torch.cat([graph_masks, grid_masks], dim=1)  # [B, final_max_len]

        x = co_embeds

        attn_weights = []
        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)
            if self.vis:
                attn_weights.append(_attn)

        x = self.transformer.norm(x)
        graph_feats, grid_feats = (
            x[:, : graph_embeds.shape[1]],
            x[:, graph_embeds.shape[1] :],
        )  # [B, max_graph_len, hid_dim], [B, max_grid_len+2, hid_dim]

        cls_feats = self.pooler(x)  # [B, hid_dim]

        ret = {
            "graph_feats": graph_feats,
            "grid_feats": grid_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "graph_masks": graph_masks,
            "grid_masks": grid_masks,
            "grid_labels": grid_labels,  # if MPP, else None
            "mo_labels": mo_labels,  # if MOC, else None
            "cif_id": cif_id,
            "attn_weights": attn_weights,
        }

        return ret

    def forward(self, batch):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Graph Grid Matching
        if "ggm" in self.current_tasks:
            ret.update(objectives.compute_ggm(self, batch))

        # MOF Topology Prediction
        if "mtp" in self.current_tasks:
            ret.update(objectives.compute_mtp(self, batch))

        # Void Fraction Prediction
        if "vfp" in self.current_tasks:
            ret.update(objectives.compute_vfp(self, batch))

        # Metal Organic Classification (or Building Block Classfication)
        if "moc" in self.current_tasks or "bbc" in self.current_tasks:
            ret.update(objectives.compute_moc(self, batch))

        # regression
        if "regression" in self.current_tasks:
            normalizer = Normalizer(self.mean, self.std, self.device)
            ret.update(objectives.compute_regression(self, batch, normalizer))

        # classification
        if "classification" in self.current_tasks:
            ret.update(objectives.compute_classification(self, batch))
        return ret

    def on_train_start(self):
        module_utils.set_task(self)
        self.write_log = True
        freeze_epochs = int(self.hparams.config.get("freeze_backbone_epochs", 0))
        if freeze_epochs > 0 and "regression" in self.current_tasks:
            self._freeze_backbone()

    def on_train_epoch_start(self):
        freeze_epochs = int(self.hparams.config.get("freeze_backbone_epochs", 0))
        if (
            self._backbone_frozen
            and freeze_epochs > 0
            and self.current_epoch >= freeze_epochs
        ):
            self._unfreeze_backbone()

    def training_step(self, batch, batch_idx):
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss

    def on_train_epoch_end(self):
        module_utils.epoch_wrapup(self)

    def on_validation_start(self):
        module_utils.set_task(self)
        self.write_log = True

    def validation_step(self, batch, batch_idx):
        output = self(batch)

    def on_validation_epoch_end(self) -> None:
        module_utils.epoch_wrapup(self)

    def on_test_start(self):
        module_utils.set_task(self)

    def test_step(self, batch, batch_idx):
        output = self(batch)
        output = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in output.items()}  # cpu for memory

        if "regression_logits" in output.keys():
            self.test_logits += output["regression_logits"].tolist()
            self.test_labels += output["regression_labels"].tolist()
        return output

    def on_test_epoch_end(self):
        module_utils.epoch_wrapup(self)

        if len(self.test_logits) > 1:
            r2 = r2_score(np.array(self.test_labels), np.array(self.test_logits))
            self.log(f"test/r2_score", r2, sync_dist=True)
            self.test_labels.clear()
            self.test_logits.clear()

    def configure_optimizers(self):
        return module_utils.set_schedule(self)

    def on_predict_start(self):
        self.write_log = False
        module_utils.set_task(self)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch)

        if "classification_logits" in output:
            if self.hparams.config["n_classes"] == 2:
                output["classification_logits_index"] = torch.round(
                    output["classification_logits"]
                ).to(torch.int)
            else:
                softmax = torch.nn.Softmax(dim=1)
                output["classification_logits"] = softmax(output["classification_logits"])
                output["classification_logits_index"] = torch.argmax(
                    output["classification_logits"], dim=1
                )

        output = {
            k: (v.cpu().tolist() if torch.is_tensor(v) else v)
            for k, v in output.items()
            if ("logits" in k) or ("labels" in k) or "cif_id" == k
        }

        return output

    def on_predict_epoch_end(self, *args):
        self.test_labels.clear()
        self.test_logits.clear()

    def on_predict_end(self):
        self.write_log = True

    def lr_scheduler_step(self, scheduler, *args):
        if len(args) == 2:
            optimizer_idx, metric = args
        elif len(args) == 1:
            (metric,) = args
        else:
            raise ValueError("lr_scheduler_step must have metric and optimizer_idx(optional)")

        if pl.__version__ >= "2.0.0":
            scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step()
