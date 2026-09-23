# MOFTransformer version 2.0.0
import os
import random
import json
import pickle

import numpy as np

import torch
from torch.nn.functional import interpolate


class Dataset(torch.utils.data.Dataset):
    """
    和 datamodules/dataset.py 基本一致，只是方便你单独实验 ALIGNN 时引用。
    """
    def __init__(
        self,
        data_dir: str,
        split: str,
        nbr_fea_len: int,
        draw_false_grid=True,
        downstream="",
        tasks=[],
    ):
        super().__init__()
        self.data_dir = data_dir
        self.draw_false_grid = draw_false_grid
        self.split = split

        assert split in {"train", "test", "val"}
        if downstream:
            path_file = os.path.join(data_dir, f"{split}_{downstream}.json")
        else:
            path_file = os.path.join(data_dir, f"{split}.json")
        print(f"read {path_file}...")

        if not os.path.isfile(path_file):
            raise FileNotFoundError(
                f"{path_file} doesn't exist. Check 'root_dataset' in config"
            )

        dict_target = json.load(open(path_file, "r"))
        self.cif_ids, self.targets = zip(*dict_target.items())

        self.nbr_fea_len = nbr_fea_len

        self.tasks = {}

        for task in tasks:
            if task in ["mtp", "vfp", "moc", "bbc"]:
                path_file = os.path.join(data_dir, f"{split}_{task}.json")
                print(f"read {path_file}...")
                assert os.path.isfile(
                    path_file
                ), f"{path_file} doesn't exist in {data_dir}"

                dict_task = json.load(open(path_file, "r"))
                cif_ids, t = zip(*dict_task.items())
                self.tasks[task] = list(t)
                assert self.cif_ids == cif_ids, print(
                    "order of keys is different in the json file"
                )

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

    def get_grid_data(self, cif_id, draw_false_grid=False):
        cell, volume, grid_data = self.get_raw_grid_data(cif_id)
        ret = {
            "cell": cell,
            "volume": volume,
            "grid_data": grid_data,
        }

        if draw_false_grid:
            random_index = random.randint(0, len(self.cif_ids) - 1)
            cif_id = self.cif_ids[random_index]
            cell, volume, grid_data = self.get_raw_grid_data(cif_id)
            ret.update(
                {
                    "false_cell": cell,
                    "fale_volume": volume,
                    "false_grid_data": grid_data,
                }
            )
        return ret

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

    def get_tasks(self, index):
        ret = dict()
        for task, value in self.tasks.items():
            ret.update({task: value[index]})
        return ret

    def __getitem__(self, index):
        ret = dict()
        cif_id = self.cif_ids[index]
        target = self.targets[index]

        ret.update(
            {
                "cif_id": cif_id,
                "target": target,
            }
        )
        ret.update(self.get_grid_data(cif_id, draw_false_grid=self.draw_false_grid))
        ret.update(self.get_graph(cif_id))
        ret.update(self.get_tasks(index))
        return ret

    @staticmethod
    def collate(batch, img_size):
        return Dataset.collate.__func__(batch, img_size)
