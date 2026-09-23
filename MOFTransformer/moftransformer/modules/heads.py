# MOFTransformer version 2.2.0
import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertPredictionHeadTransform,
)


class Pooler(nn.Module):
    def __init__(self, hidden_size, index=0):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.index = index

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, self.index]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GGMHead(nn.Module):
    """
    head for Graph Grid Matching
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MPPHead(nn.Module):
    """
    head for Masked Patch Prediction (regression version)
    """

    def __init__(self, hid_dim):
        super().__init__()

        bert_config = BertConfig(
            hidden_size=hid_dim,
        )
        self.transform = BertPredictionHeadTransform(bert_config)
        self.decoder = nn.Linear(hid_dim, 101 + 2)  # bins

    def forward(self, x):  # [B, max_len, hid_dim]
        x = self.transform(x)  # [B, max_len, hid_dim]
        x = self.decoder(x)  # [B, max_len, bins]
        return x


class MTPHead(nn.Module):
    """
    head for MOF Topology Prediction
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.fc = nn.Linear(hid_dim, 1397)  # len(assets/topology.json)

    def forward(self, x):
        x = self.fc(x)
        return x


class VFPHead(nn.Module):
    """
    head for Void Fraction Prediction
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(hid_dim)
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):
        x = self.bn(x)
        x = self.fc(x)
        return x


class RegressionHead(nn.Module):
    """
    head for Regression
    """

    def __init__(self, hid_dim, n_targets=1):
        super().__init__()
        self.fc = nn.Linear(hid_dim, n_targets)

    def forward(self, x):
        x = self.fc(x)
        return x


class FusionRegressionHead(nn.Module):
    """
    Regression head that explicitly combines global CLS, graph, and grid summaries.
    """

    def __init__(self, hid_dim, n_targets=1, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hid_dim * 3)
        self.fc1 = nn.Linear(hid_dim * 3, hid_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hid_dim, n_targets)

    @staticmethod
    def _masked_mean(feats, masks):
        masks = masks.to(feats.dtype).unsqueeze(-1)
        denom = masks.sum(dim=1).clamp_min(1.0)
        return (feats * masks).sum(dim=1) / denom

    def forward(self, infer):
        graph_feats = infer["graph_feats"]
        graph_masks = infer["graph_masks"]
        if graph_feats.shape[1] > 1:
            graph_feats = graph_feats[:, 1:]
            graph_masks = graph_masks[:, 1:]

        graph_pool = self._masked_mean(graph_feats, graph_masks)
        grid_pool = self._masked_mean(infer["grid_feats"], infer["grid_masks"])
        x = torch.cat([infer["cls_feats"], graph_pool, grid_pool], dim=-1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)


class GatedResidualRegressionHead(nn.Module):
    """
    Lightweight regression head:
    use graph/grid pooled features only as gated residual corrections to CLS.
    This is intentionally lower-risk than full concatenation fusion.
    """

    def __init__(self, hid_dim, n_targets=1, dropout=0.1):
        super().__init__()
        self.graph_proj = nn.Linear(hid_dim, hid_dim)
        self.grid_proj = nn.Linear(hid_dim, hid_dim)
        self.gate = nn.Linear(hid_dim * 3, 2)
        self.norm = nn.LayerNorm(hid_dim)
        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hid_dim, n_targets)

    @staticmethod
    def _masked_mean(feats, masks):
        masks = masks.to(feats.dtype).unsqueeze(-1)
        denom = masks.sum(dim=1).clamp_min(1.0)
        return (feats * masks).sum(dim=1) / denom

    def forward(self, infer):
        graph_feats = infer["graph_feats"]
        graph_masks = infer["graph_masks"]
        if graph_feats.shape[1] > 1:
            graph_feats = graph_feats[:, 1:]
            graph_masks = graph_masks[:, 1:]

        graph_pool = self._masked_mean(graph_feats, graph_masks)
        grid_pool = self._masked_mean(infer["grid_feats"], infer["grid_masks"])
        cls_feats = infer["cls_feats"]

        gate_logits = self.gate(torch.cat([cls_feats, graph_pool, grid_pool], dim=-1))
        gates = torch.sigmoid(gate_logits)

        residual = (
            gates[:, 0:1] * self.graph_proj(graph_pool)
            + gates[:, 1:2] * self.grid_proj(grid_pool)
        )
        x = self.norm(cls_feats + residual)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)


class ModalRegressionHead(nn.Module):
    """
    Flexible regression head that can read from different modality summaries.
    Intended for auxiliary descriptor tasks where chemistry and pore descriptors
    may depend on different sources.
    """

    def __init__(self, hid_dim, n_targets=1, source="cls", dropout=0.1):
        super().__init__()
        valid_sources = {
            "cls",
            "graph",
            "grid",
            "cls_graph",
            "cls_grid",
            "graph_grid",
            "cls_graph_grid",
        }
        if source not in valid_sources:
            raise ValueError(f"Unknown ModalRegressionHead source: {source}")
        self.source = source
        n_inputs = len(source.split("_"))
        self.norm = nn.LayerNorm(hid_dim * n_inputs)
        self.fc1 = nn.Linear(hid_dim * n_inputs, hid_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hid_dim, n_targets)

    @staticmethod
    def _masked_mean(feats, masks):
        masks = masks.to(feats.dtype).unsqueeze(-1)
        denom = masks.sum(dim=1).clamp_min(1.0)
        return (feats * masks).sum(dim=1) / denom

    def forward(self, infer):
        parts = []
        if "cls" in self.source:
            parts.append(infer["cls_feats"])

        if "graph" in self.source:
            graph_feats = infer["graph_feats"]
            graph_masks = infer["graph_masks"]
            if graph_feats.shape[1] > 1:
                graph_feats = graph_feats[:, 1:]
                graph_masks = graph_masks[:, 1:]
            parts.append(self._masked_mean(graph_feats, graph_masks))

        if "grid" in self.source:
            parts.append(self._masked_mean(infer["grid_feats"], infer["grid_masks"]))

        x = torch.cat(parts, dim=-1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)


class DescriptorFusionRegressionHead(nn.Module):
    """
    Downstream regression head that fuses transformer CLS features with
    explicit descriptor vectors (e.g. selected ZEOPP descriptors).
    """

    def __init__(self, hid_dim, descriptor_dim, n_targets=1, dropout=0.1):
        super().__init__()
        if descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be > 0 for DescriptorFusionRegressionHead")
        self.norm = nn.LayerNorm(hid_dim + descriptor_dim)
        self.fc1 = nn.Linear(hid_dim + descriptor_dim, hid_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hid_dim, n_targets)

    def forward(self, infer, descriptors):
        if not torch.is_tensor(descriptors):
            descriptors = torch.tensor(
                descriptors,
                device=infer["cls_feats"].device,
                dtype=infer["cls_feats"].dtype,
            )
        else:
            descriptors = descriptors.to(
                device=infer["cls_feats"].device,
                dtype=infer["cls_feats"].dtype,
            )
        x = torch.cat([infer["cls_feats"], descriptors], dim=-1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)


class DescriptorModalFusionRegressionHead(nn.Module):
    """
    Stronger downstream fusion head:
    combine CLS, graph pool, grid pool, and explicit descriptor vectors.
    """

    def __init__(self, hid_dim, descriptor_dim, n_targets=1, dropout=0.1):
        super().__init__()
        if descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be > 0 for DescriptorModalFusionRegressionHead")
        self.norm = nn.LayerNorm(hid_dim * 3 + descriptor_dim)
        self.fc1 = nn.Linear(hid_dim * 3 + descriptor_dim, hid_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hid_dim, n_targets)

    @staticmethod
    def _masked_mean(feats, masks):
        masks = masks.to(feats.dtype).unsqueeze(-1)
        denom = masks.sum(dim=1).clamp_min(1.0)
        return (feats * masks).sum(dim=1) / denom

    def forward(self, infer, descriptors):
        graph_feats = infer["graph_feats"]
        graph_masks = infer["graph_masks"]
        if graph_feats.shape[1] > 1:
            graph_feats = graph_feats[:, 1:]
            graph_masks = graph_masks[:, 1:]
        graph_pool = self._masked_mean(graph_feats, graph_masks)
        grid_pool = self._masked_mean(infer["grid_feats"], infer["grid_masks"])

        if not torch.is_tensor(descriptors):
            descriptors = torch.tensor(
                descriptors,
                device=infer["cls_feats"].device,
                dtype=infer["cls_feats"].dtype,
            )
        else:
            descriptors = descriptors.to(
                device=infer["cls_feats"].device,
                dtype=infer["cls_feats"].dtype,
            )

        x = torch.cat([infer["cls_feats"], graph_pool, grid_pool, descriptors], dim=-1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)


class DescriptorResidualRegressionHead(nn.Module):
    """
    Conservative descriptor-aware head:
    keep CLS as the primary predictor and use descriptors only as a gated residual.
    """

    def __init__(self, hid_dim, descriptor_dim, n_targets=1, dropout=0.1):
        super().__init__()
        if descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be > 0 for DescriptorResidualRegressionHead")
        self.base = nn.Linear(hid_dim, n_targets)
        self.desc_norm = nn.LayerNorm(descriptor_dim)
        self.desc_fc1 = nn.Linear(descriptor_dim, hid_dim)
        self.desc_act = nn.GELU()
        self.desc_dropout = nn.Dropout(dropout)
        self.desc_fc2 = nn.Linear(hid_dim, n_targets)
        self.gate = nn.Linear(hid_dim + descriptor_dim, n_targets)

    def forward(self, infer, descriptors):
        cls_feats = infer["cls_feats"]
        if not torch.is_tensor(descriptors):
            descriptors = torch.tensor(
                descriptors,
                device=cls_feats.device,
                dtype=cls_feats.dtype,
            )
        else:
            descriptors = descriptors.to(device=cls_feats.device, dtype=cls_feats.dtype)

        base = self.base(cls_feats)
        desc = self.desc_norm(descriptors)
        desc = self.desc_fc1(desc)
        desc = self.desc_act(desc)
        desc = self.desc_dropout(desc)
        desc = self.desc_fc2(desc)
        gate = torch.sigmoid(self.gate(torch.cat([cls_feats, descriptors], dim=-1)))
        return base + gate * desc


class ClassificationHead(nn.Module):
    """
    head for Classification
    """

    def __init__(self, hid_dim, n_classes):
        super().__init__()

        if n_classes == 2:
            self.fc = nn.Linear(hid_dim, 1)
            self.binary = True
        else:
            self.fc = nn.Linear(hid_dim, n_classes)
            self.binary = False

    def forward(self, x):
        x = self.fc(x)

        return x, self.binary


class MOCHead(nn.Module):
    """
    head for Metal Organic Classification
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):
        """
        :param x: graph_feats [B, graph_len, hid_dim]
        :return: [B, graph_len]
        """
        x = self.fc(x)  # [B, graph_len, 1]
        x = x.squeeze(dim=-1)  # [B, graph_len]
        return x
