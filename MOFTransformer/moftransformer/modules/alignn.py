import random
import torch
import torch.nn as nn


class EdgeGatedConv(nn.Module):
    """
    Edge-gated graph convolution.
    用于:
      - 原子图:   节点 = atom, 边 = bond
      - 线图:     节点 = bond, 边 = angle
    输入 node_feats / edge_feats 维度一致 = hidden_dim
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.edge_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.edge_update_mlp = nn.Linear(3 * hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, node_feats: torch.Tensor, edge_index, edge_feats: torch.Tensor):
        """
        node_feats: [N, hidden_dim]
        edge_index: (src, dst) where src/dst: [E]
        edge_feats: [E, hidden_dim]
        """
        src, dst = edge_index  # [E], [E]

        # gate = sigmoid(W_e * e)
        gate = torch.sigmoid(self.edge_mlp(edge_feats))  # [E, hidden_dim]

        # message = gate * W_n * h_src
        msg = self.node_mlp(node_feats[src]) * gate  # [E, hidden_dim]

        # 聚合到 dst
        N = node_feats.shape[0]
        # 在 AMP 下，线性层输出可能是 fp16/bf16，这里需要和 msg 保持同 dtype，
        # 否则 index_add_ 会因为 self/source 类型不一致而报错。
        agg = torch.zeros((N, self.hidden_dim), device=node_feats.device, dtype=msg.dtype)
        agg.index_add_(0, dst, msg)
        agg = self.dropout(agg)
        if agg.dtype != node_feats.dtype:
            agg = agg.to(node_feats.dtype)

        new_node = node_feats + self.act(agg)

        # 更新边：结合 new_node[src] / new_node[dst] / old_edge
        h_src = new_node[src]
        h_dst = new_node[dst]
        edge_input = torch.cat([h_src, h_dst, edge_feats], dim=-1)
        delta_e = self.edge_update_mlp(edge_input)
        delta_e = self.dropout(self.act(delta_e))
        new_edge = edge_feats + delta_e

        return new_node, new_edge


class ALIGNNBlock(nn.Module):
    """
    一个 ALIGNN Block：
      先在线图上更新 (bond, angle)，再在原子图上更新 (atom, bond)
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.line_conv = EdgeGatedConv(hidden_dim, dropout=dropout)
        self.atom_conv = EdgeGatedConv(hidden_dim, dropout=dropout)

    def forward(
        self,
        atom_feats: torch.Tensor,
        bond_feats: torch.Tensor,
        angle_feats: torch.Tensor,
        atom_edge_index,
        angle_edge_index,
    ):
        # 线图: 节点 = bond_feats, 边 = angle_feats
        bond_feats, angle_feats = self.line_conv(
            node_feats=bond_feats, edge_index=angle_edge_index, edge_feats=angle_feats
        )
        # 原子图: 节点 = atom_feats, 边 = bond_feats
        atom_feats, bond_feats = self.atom_conv(
            node_feats=atom_feats, edge_index=atom_edge_index, edge_feats=bond_feats
        )
        return atom_feats, bond_feats, angle_feats


class ALIGNNGraphEmbeddings(nn.Module):
    """
    用 ALIGNN 结构代替 CGCNN 的 GraphEmbeddings，接口保持基本兼容：

    forward 输入：
        atom_num: [N_total]
        nbr_idx:  [N_total, M]
        nbr_fea:  [N_total, M, nbr_fea_len]
        crystal_atom_idx: List[Tensor] 长度 B，每个是该结构在 batch 中的节点索引
        uni_idx: List[List[List[int]]] 每个结构内拓扑等价原子索引组
        uni_count: List[int]
        moc: 可选金属标签（原版 MOFTransformer 用于 MOC 任务）

        angle_src: [A_total]  (edge-id 范围 [0, N_total*M))
        angle_dst: [A_total]
        angle_fea: [A_total, angle_fea_len]

    forward 输出：
        new_atom_fea: [B, max_graph_len, out_dim]
        mask:         [B, max_graph_len]
        mo_label:     [B, max_graph_len]
    """

    def __init__(
        self,
        in_feats: int = 119,
        max_graph_len: int = 256,
        out_dim: int = 384,
        num_layers: int = 4,
        dropout: float = 0.1,
        vis: bool = False,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.max_graph_len = max_graph_len
        self.hidden_dim = out_dim  # 内部 hidden_dim = 输出 dim，方便和 Transformer 对齐
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.vis = vis

        # 原子类型 embedding (原子序数 → hidden_dim)
        self.atom_embedding = nn.Embedding(in_feats, self.hidden_dim)

        # 这两个 MLP 在第一次 forward 时，按真实输入维度 lazy 初始化
        self.radial_mlp = None    # nbr_fea → hidden_dim
        self.angle_mlp = None     # angle_fea → hidden_dim

        # ALIGNN blocks
        self.blocks = nn.ModuleList(
            [ALIGNNBlock(hidden_dim=self.hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )

    def _init_edge_mlps(self, nbr_fea: torch.Tensor, angle_fea: torch.Tensor):
        """
        根据实际的 nbr_fea_dim / angle_fea_dim lazy 初始化 MLP，
        避免你在 config 里再加一堆超参。
        """
        device = nbr_fea.device
        if self.radial_mlp is None:
            in_dim = nbr_fea.shape[-1]
            self.radial_mlp = nn.Linear(in_dim, self.hidden_dim).to(device)
        if self.angle_mlp is None and angle_fea is not None and angle_fea.numel() > 0:
            in_dim = angle_fea.shape[-1]
            self.angle_mlp = nn.Linear(in_dim, self.hidden_dim).to(device)

    def reconstruct_batch(self, atom_fea, crystal_atom_idx, uni_idx, uni_count, moc):
        """
        完全照搬 CGCNN 版 GraphEmbeddings 的重构逻辑：
        - 按拓扑等价原子分组，从每组随机选一个代表
        - 打乱顺序，截断到 max_graph_len
        - 输出 [B, max_graph_len, hid_dim] 和 mask
        """
        batch_size = len(crystal_atom_idx)

        new_atom_fea = torch.full(
            size=[batch_size, self.max_graph_len, self.out_dim], fill_value=0.0, device=atom_fea.device
        )

        mo_label = torch.full(
            size=[batch_size, self.max_graph_len], fill_value=-100.0, device=atom_fea.device
        )

        for bi, c_atom_idx in enumerate(crystal_atom_idx):
            # set uni_idx with (descending count or random) and cut max_graph_len
            idx_ = torch.LongTensor([random.choice(u) for u in uni_idx[bi]])[
                : self.max_graph_len
            ]
            rand_idx = idx_[torch.randperm(len(idx_))]
            if self.vis:
                rand_idx = idx_
            new_atom_fea[bi][: len(rand_idx)] = atom_fea[c_atom_idx][rand_idx]

            if moc:
                mo = torch.zeros(len(c_atom_idx), device=atom_fea.device)
                metal_idx = moc[bi]
                mo[metal_idx] = 1
                mo_label[bi][: len(rand_idx)] = mo[rand_idx]

        mask = (new_atom_fea.sum(dim=-1) != 0).float()

        return new_atom_fea, mask, mo_label

    def forward(
        self,
        atom_num,
        nbr_idx,
        nbr_fea,
        crystal_atom_idx,
        uni_idx,
        uni_count,
        moc=None,
        angle_src=None,
        angle_dst=None,
        angle_fea=None,
    ):
        """
        主入口：给 Module.infer 调用。
        """
        device = atom_num.device
        N, M = nbr_idx.shape
        E = N * M

        # Edge index for atom graph: edges from center -> neighbor
        src = torch.arange(N, device=device).unsqueeze(1).expand(N, M).reshape(-1)  # [E]
        dst = nbr_idx.reshape(-1)  # [E]

        # lazy 初始化 edge MLP
        if angle_fea is None:
            # 没有角度信息时，也初始化 radial_mlp
            self._init_edge_mlps(nbr_fea.reshape(-1, nbr_fea.shape[-1]), None)
        else:
            self._init_edge_mlps(nbr_fea.reshape(-1, nbr_fea.shape[-1]), angle_fea)

        # 原子初始特征
        atom_feats = self.atom_embedding(atom_num)  # [N, hidden_dim]

        # 边初始特征（径向 RBF + MLP）：直接用你的 nbr_fea
        bond_feats = self.radial_mlp(nbr_fea.reshape(-1, nbr_fea.shape[-1]))  # [E, hidden_dim]

        if angle_fea is not None and angle_fea.numel() > 0:
            angle_feats = self.angle_mlp(angle_fea)  # [A, hidden_dim]
            angle_edge_index = (angle_src.to(device), angle_dst.to(device))  # [A], [A]
        else:
            # 没有角度信息时，构造一个空图，模型会退化为“GraphConv 版”
            angle_feats = torch.zeros((0, self.hidden_dim), device=device, dtype=atom_feats.dtype)
            angle_edge_index = (
                torch.zeros((0,), device=device, dtype=torch.long),
                torch.zeros((0,), device=device, dtype=torch.long),
            )

        atom_edge_index = (src, dst)

        # ALIGNN blocks
        for blk in self.blocks:
            atom_feats, bond_feats, angle_feats = blk(
                atom_feats=atom_feats,
                bond_feats=bond_feats,
                angle_feats=angle_feats,
                atom_edge_index=atom_edge_index,
                angle_edge_index=angle_edge_index,
            )

        # 此时 atom_feats: [N, hidden_dim] = out_dim
        new_atom_fea, mask, mo_label = self.reconstruct_batch(
            atom_fea=atom_feats,
            crystal_atom_idx=crystal_atom_idx,
            uni_idx=uni_idx,
            uni_count=uni_count,
            moc=moc,
        )
        return new_atom_fea, mask, mo_label
