import os
import json
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import interpolate


class CIFOnlyDataset(Dataset):
    """
    只用 CIF / graph / grid 做推理，不包含 target。
    结构和 datamodules/dataset.py 基本一致，方便直接喂给 Module.infer。
    """

    def __init__(self, data_dir, nbr_fea_len=100, img_size=30, split="test"):
        self.data_dir = data_dir
        self.img_size = img_size
        self.nbr_fea_len = nbr_fea_len
        self.split = split

        # 根据 .graphdata 列出所有 cif_id
        split_dir = os.path.join(data_dir, split)
        self.cif_ids = [
            f.split(".")[0] for f in os.listdir(split_dir) if f.endswith(".graphdata")
        ]

    def __len__(self):
        return len(self.cif_ids)

    @staticmethod
    def make_grid_data(grid_data, emin=-5000.0, emax=5000, bins=101):
        grid_data[grid_data <= emin] = emin
        grid_data[grid_data > emax] = emax
        x = np.linspace(emin, emax, bins)
        new_grid_data = np.digitize(grid_data, x) + 1
        return new_grid_data

    @staticmethod
    def calculate_volume(a, b, c, angle_a, angle_b, angle_c):
        a_ = np.cos(angle_a * np.pi / 180)
        b_ = np.cos(angle_b * np.pi / 180)
        c_ = np.cos(angle_c * np.pi / 180)
        v = a * b * c * np.sqrt(1 - a_**2 - b_**2 - c_**2 + 2 * a_ * b_ * c_)
        return v.item() / (60 * 60 * 60)  # normalized volume

    def get_raw_grid_data(self, cif_id):
        file_grid = os.path.join(self.data_dir, self.split, f"{cif_id}.grid")
        file_griddata = os.path.join(self.data_dir, self.split, f"{cif_id}.griddata16")

        with open(file_grid, "r") as f:
            lines = f.readlines()
            a, b, c = [float(i) for i in lines[0].split()[1:]]
            angle_a, angle_b, angle_c = [float(i) for i in lines[1].split()[1:]]
            cell = [int(i) for i in lines[2].split()[1:]]

        volume = self.calculate_volume(a, b, c, angle_a, angle_b, angle_c)

        grid_data = pickle.load(open(file_griddata, "rb"))
        grid_data = self.make_grid_data(grid_data)
        grid_data = torch.FloatTensor(grid_data)

        return cell, volume, grid_data

    @staticmethod
    def get_gaussian_distance(distances, num_step, dmax, dmin=0, var=0.2):
        assert dmin < dmax
        _filter = np.linspace(dmin, dmax, num_step)
        return np.exp(-((distances[..., np.newaxis] - _filter) ** 2) / var**2).float()

    def get_graph(self, cif_id):
        file_graph = os.path.join(self.data_dir, self.split, f"{cif_id}.graphdata")
        graphdata = pickle.load(open(file_graph, "rb"))

        atom_num = torch.LongTensor(graphdata[1].copy())
        nbr_idx = torch.LongTensor(graphdata[2].copy()).view(len(atom_num), -1)
        nbr_dist = torch.FloatTensor(graphdata[3].copy()).view(len(atom_num), -1)

        nbr_fea = torch.FloatTensor(
            self.get_gaussian_distance(nbr_dist, num_step=self.nbr_fea_len, dmax=8)
        )

        uni_idx = graphdata[4]
        uni_count = graphdata[5]

        angle_src = None
        angle_dst = None
        angle_fea = None
        if len(graphdata) >= 9:
            angle_src = torch.LongTensor(graphdata[6].copy())
            angle_dst = torch.LongTensor(graphdata[7].copy())
            angle_fea = torch.FloatTensor(graphdata[8].copy())

        ret = {
            "atom_num": atom_num,
            "nbr_idx": nbr_idx,
            "nbr_fea": nbr_fea,
            "uni_idx": uni_idx,
            "uni_count": uni_count,
        }
        if angle_src is not None and angle_fea is not None:
            ret.update(
                {
                    "angle_src": angle_src,
                    "angle_dst": angle_dst,
                    "angle_fea": angle_fea,
                }
            )
        return ret

    def __getitem__(self, index):
        cif_id = self.cif_ids[index]

        ret = {"cif_id": cif_id}

        cell, volume, grid_data = self.get_raw_grid_data(cif_id)
        ret.update(
            {
                "cell": cell,
                "volume": volume,
                "grid_data": grid_data,
            }
        )
        ret.update(self.get_graph(cif_id))
        return ret

    @staticmethod
    def collate(batch, img_size):
        """
        基本复制 datamodules/dataset.py 的 collate，只是不包含 target/task。
        """
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        # graph
        batch_atom_num = dict_batch["atom_num"]
        batch_nbr_idx = dict_batch["nbr_idx"]
        batch_nbr_fea = dict_batch["nbr_fea"]

        batch_angle_src = dict_batch.get("angle_src", None)
        batch_angle_dst = dict_batch.get("angle_dst", None)
        batch_angle_fea = dict_batch.get("angle_fea", None)

        crystal_atom_idx = []
        base_idx = 0
        edge_base = 0

        for i, nbr_idx in enumerate(batch_nbr_idx):
            n_i = nbr_idx.shape[0]
            m_i = nbr_idx.shape[1]

            crystal_atom_idx.append(torch.arange(n_i) + base_idx)
            nbr_idx += base_idx
            base_idx += n_i

            if batch_angle_src is not None and batch_angle_src[i] is not None:
                e_i = n_i * m_i
                batch_angle_src[i] = batch_angle_src[i] + edge_base
                batch_angle_dst[i] = batch_angle_dst[i] + edge_base
                edge_base += e_i

        dict_batch["atom_num"] = torch.cat(batch_atom_num, dim=0)
        dict_batch["nbr_idx"] = torch.cat(batch_nbr_idx, dim=0)
        dict_batch["nbr_fea"] = torch.cat(batch_nbr_fea, dim=0)
        dict_batch["crystal_atom_idx"] = crystal_atom_idx

        if batch_angle_src is not None and batch_angle_src[0] is not None:
            dict_batch["angle_src"] = torch.cat(batch_angle_src, dim=0)
            dict_batch["angle_dst"] = torch.cat(batch_angle_dst, dim=0)
            dict_batch["angle_fea"] = torch.cat(batch_angle_fea, dim=0)

        # grid
        batch_grid_data = dict_batch["grid_data"]
        batch_cell = dict_batch["cell"]
        new_grids = []
        for bi in range(batch_size):
            orig = batch_grid_data[bi].view(batch_cell[bi][::-1]).transpose(0, 2)
            if batch_cell[bi] == [img_size, img_size, img_size]:
                orig = orig[None, None, :, :, :]
            else:
                orig = interpolate(
                    orig[None, None, :, :, :],
                    size=[img_size, img_size, img_size],
                    mode="trilinear",
                    align_corners=True,
                )
            new_grids.append(orig)
        new_grids = torch.concat(new_grids, axis=0)
        dict_batch["grid"] = new_grids

        dict_batch.pop("grid_data", None)
        dict_batch.pop("cell", None)

        return dict_batch
