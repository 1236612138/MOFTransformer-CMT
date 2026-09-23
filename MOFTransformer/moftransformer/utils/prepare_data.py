# Version 2.2.0
import os
import math
import sys
import time
import warnings
import logging
import logging.handlers
import json
import subprocess
import hashlib
import pickle
import shutil
from multiprocessing import Pool
from pathlib import Path
from collections import namedtuple
from collections.abc import Iterable

import numpy as np

from tqdm import tqdm
from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cssr import Cssr

from ase.io import read
from ase.neighborlist import natural_cutoffs
from ase import neighborlist
from ase.build import make_supercell

from moftransformer import __root_dir__

GRIDAY_PATH = os.path.join(__root_dir__, "libs/GRIDAY/scripts/grid_gen")
FF_PATH = os.path.join(__root_dir__, "libs/GRIDAY/FF")

# 角度 RBF 的基函数个数（与 ALIGNN 中 angle_fea_len 对应）
DEFAULT_NUM_ANGLE_BASIS = 8


def _suppress_noisy_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*We strongly encourage explicit `encoding`.*",
        category=Warning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*get_structures is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Issues encountered while parsing CIF.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Some occupancies.*sum to > 1.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*No structure parsed for section.*",
        category=UserWarning,
    )


def get_logger(filename):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_unique_atoms(atoms):
    # get graph
    cutoff = natural_cutoffs(atoms)
    neighbor_list = neighborlist.NeighborList(
        cutoff, self_interaction=True, bothways=True
    )
    neighbor_list.update(atoms)
    matrix = neighbor_list.get_connectivity_matrix()

    # Get N, N^2
    numbers = atoms.numbers
    number_sqr = np.multiply(numbers, numbers)

    matrix_sqr = matrix.dot(matrix)
    matrix_cub = matrix_sqr.dot(matrix)
    matrix_sqr.data[:] = 1  # count 1 for atoms

    # calculate
    list_n = [numbers, number_sqr]
    list_m = [matrix, matrix_sqr, matrix_cub]

    arr = [numbers]

    for m in list_m:
        for n in list_n:
            arr.append(m.dot(n))

    arr = np.vstack(arr).transpose()

    uni, unique_idx, unique_count = np.unique(
        arr, axis=0, return_index=True, return_counts=True
    )

    # sort
    final_uni = uni[np.argsort(-unique_count)].tolist()
    final_unique_count = unique_count[np.argsort(-unique_count)].tolist()

    arr = arr.tolist()
    final_unique_idx = []
    for u in final_uni:
        final_unique_idx.append([i for i, a in enumerate(arr) if a == u])

    return final_unique_idx, final_unique_count


def _angle_rbf(cos_theta: float, num_basis: int) -> np.ndarray:
    """
    对 cos(theta) ∈ [-1, 1] 做 RBF 展开，得到长度 num_basis 的向量。
    """
    centers = np.linspace(-1.0, 1.0, num_basis, dtype=np.float32)
    if num_basis > 1:
        width = (centers[1] - centers[0])
    else:
        width = 2.0
    feats = np.exp(-((cos_theta - centers) ** 2) / (2.0 * (width ** 2)))
    return feats.astype(np.float32)


def get_crystal_graph(
    atoms,
    radius=8,
    max_num_nbr=12,
    num_angle_basis: int = DEFAULT_NUM_ANGLE_BASIS,
):
    """
    生成图结构 + 角度信息.

    返回：
        atom_num: np.int8 [N]
        nbr_idx: np.int16 [N * max_num_nbr]       （按行展开）
        nbr_dist: np.float32 [N * max_num_nbr]    （按行展开）
        uni_idx: List[List[int]]                  （拓扑等价原子）
        uni_count: np.int16 [n_unique]
        angle_src: np.int32 [A]                   （line graph 边的源 edge-id）
        angle_dst: np.int32 [A]                   （line graph 边的目标 edge-id）
        angle_fea: np.float32 [A, num_angle_basis]（角度 RBF 展开）
    """
    # 所有原子对距离矩阵
    dist_mat = atoms.get_all_distances(mic=True)  # [N, N]
    # 把自距离(0)替换为一个很大的数，这样在选最近邻时不会选到自己
    nbr_mat = np.where(dist_mat > 0, dist_mat, 1000.0)

    N = len(atoms)
    # 每个原子选 max_num_nbr 个最近邻
    nbr_idx_mat = np.zeros((N, max_num_nbr), dtype=np.int32)
    nbr_dist_mat = np.zeros((N, max_num_nbr), dtype=np.float32)

    for i, row in enumerate(nbr_mat):
        idx = np.argsort(row)[:max_num_nbr]
        nbr_idx_mat[i] = idx
        nbr_dist_mat[i] = row[idx]

    # ===== 角度信息：对每个中心原子 i，在它的 M 个邻居中两两组合，构造角度 =====
    angle_src_list = []
    angle_dst_list = []
    angle_fea_list = []

    for i in range(N):
        # 中心原子 i 的邻居
        neigh = nbr_idx_mat[i]  # [max_num_nbr]
        dist_i = nbr_dist_mat[i]  # [max_num_nbr]

        for a in range(max_num_nbr):
            j = int(neigh[a])
            dij = float(dist_i[a])
            if j == i or dij <= 1e-6:
                continue
            for b in range(a + 1, max_num_nbr):
                k = int(neigh[b])
                dik = float(dist_i[b])
                if k == i or dik <= 1e-6:
                    continue

                djk = float(dist_mat[j, k])
                if djk <= 0.0:
                    continue

                # 余弦定理: cosθ = (dij^2 + dik^2 - djk^2) / (2 * dij * dik)
                cos_theta = (dij ** 2 + dik ** 2 - djk ** 2) / (2.0 * dij * dik)
                cos_theta = max(-1.0, min(1.0, cos_theta))  # 数值稳定

                angle_fea = _angle_rbf(cos_theta, num_angle_basis)
                # 边的 id: 对应 i 的第 a / b 个邻居
                e1 = i * max_num_nbr + a
                e2 = i * max_num_nbr + b

                # 为了让 line graph 是有向双向边，这里正反都加
                angle_src_list.append(e1)
                angle_dst_list.append(e2)
                angle_fea_list.append(angle_fea)

                angle_src_list.append(e2)
                angle_dst_list.append(e1)
                angle_fea_list.append(angle_fea)

    # get same-topo atoms
    uni_idx, uni_count = get_unique_atoms(atoms)

    # convert to small size
    atom_num = np.array(list(atoms.numbers), dtype=np.int8)
    nbr_idx = nbr_idx_mat.reshape(-1).astype(np.int16)
    nbr_dist = nbr_dist_mat.reshape(-1).astype(np.float32)
    uni_count = np.array(uni_count, dtype=np.int16)

    if len(angle_fea_list) > 0:
        angle_src = np.array(angle_src_list, dtype=np.int32)
        angle_dst = np.array(angle_dst_list, dtype=np.int32)
        angle_fea = np.stack(angle_fea_list).astype(np.float32)
    else:
        angle_src = np.zeros((0,), dtype=np.int32)
        angle_dst = np.zeros((0,), dtype=np.int32)
        angle_fea = np.zeros((0, num_angle_basis), dtype=np.float32)

    return atom_num, nbr_idx, nbr_dist, uni_idx, uni_count, angle_src, angle_dst, angle_fea


def _calculate_scaling_matrix_for_orthogonal_supercell(cell_matrix, eps=0.01):
    """
    cell_matrix: contains lattice vector as column vectors.
                 e.g. cell_matrix[:, 0] = a.
    eps: when value < eps, the value is assumed as zero.
    """
    inv = np.linalg.inv(cell_matrix)

    # Get minimum absolute values of each row.
    abs_inv = np.abs(inv)
    mat = np.where(abs_inv < eps, np.full_like(abs_inv, 1e30), abs_inv)
    min_values = np.min(mat, axis=1)

    # Normalize each row with minimum absolute value of each row.
    normed_inv = inv / min_values[:, np.newaxis]

    # Calculate scaling_matrix.
    # New cell = np.dot(scaling_matrix, cell_matrix).
    scaling_matrix = np.around(normed_inv).astype(np.int32)

    return scaling_matrix


def make_float16_griddata(file_griddata):
    griddata = np.fromfile(file_griddata, dtype=np.float32)
    griddata[griddata > 6e4] = 6e4
    griddata[griddata < -6e4] = -6e4
    griddata = griddata.astype(np.float16)
    return griddata


def get_energy_grid(atoms, cif_id, root_dataset, eg_logger):
    # Before 1.1.1 version : num_grid = [str(round(cell)) for cell in structure.lattice.abc]
    # After 1.1.1 version : num_grid = [30, 30, 30]
    global GRIDAY_PATH, FF_PATH

    eg_file = os.path.join(root_dataset, cif_id)
    random_str = str(np.random.rand()).encode()
    tmp_file = os.path.join(
        root_dataset, f"{hashlib.sha256(random_str).hexdigest()}.cssr"
    )

    try:
        structure = AseAtomsAdaptor().get_structure(atoms)
        Cssr(structure).write_file(tmp_file)
        num_grid = ["30", "30", "30"]
        proc = subprocess.Popen(
            [
                GRIDAY_PATH,
                *num_grid,
                f"{FF_PATH}/UFF_Type.def",
                f"{FF_PATH}/UFF_FF.def",
                tmp_file,
                eg_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = proc.communicate()
    finally:
        # remove temp_file
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    if err:
        eg_logger.info(f"{cif_id} energy grid failed {err}")
        return False
    else:
        eg_logger.info(f"{cif_id} energy grid success")

    if os.path.exists(eg_file + ".griddata"):
        grid_data = make_float16_griddata(eg_file + ".griddata")
        path_save = os.path.join(root_dataset, f"{cif_id}.griddata16")
        pickle.dump(grid_data, open(path_save, "wb"))
        eg_logger.info(f"{cif_id} energy grid changed to np16")

        try:
            os.remove(eg_file + ".griddata")
        except Exception as e:
            print(e)
        return True
    else:
        eg_logger.info(f"{cif_id} energy grid failed to change to np16")
        return False


def _split_dataset(root_dataset: Path, **kwargs):
    """
    make train_{task}.json, test_{task}.json, and val_{task}.json from raw_{task}.json
    :param root_cifs: root for cif files
    :param root_dataset: root for generated datasets
    :param single_task: name of downstream tasks
    :param kwargs:
        - overwrite_json (bool) : If True, overwrite {split}_task.json file when it exists. (default : False)
        - seed : (int) random seed for split data. (default : 42)
        - duplicate : (bool) If True, allow duplication of data in train, test, and validation. (default: False)
        - train_fraction : (float) fraction for train dataset. train_fraction + test_fraction must be smaller than 1 (default : 0.8)
        - test_fraction : (float) fraction for test dataset. train_fraction + test_fraction must be smaller than 1 (default : 0.1)

    :return:
    """
    # get argument from kwargs
    seed = kwargs.get("seed", 42)
    threshold = kwargs.get("threshold", 0.01)
    train_fraction = kwargs.get("train_fraction", 0.8)
    test_fraction = kwargs.get("test_fraction", 0.1)

    # get directories
    total_dir = root_dataset / "total"
    assert total_dir.exists()

    split_dir = {split: root_dataset / split for split in ["train", "test", "val"]}
    for direc in split_dir.values():
        direc.mkdir(exist_ok=True)

    # get success prepare-data list
    cif_list = {}
    CifPath = namedtuple("CifPath", ["cif", "graphdata", "grid", "griddata16"])
    for cif in total_dir.glob("*.cif"):
        cif_id = cif.stem
        graphdata = cif.with_suffix(".graphdata")
        grid = cif.with_suffix(".grid")
        griddata = cif.with_suffix(".griddata16")

        if cif.exists() and graphdata.exists() and grid.exists() and griddata.exists():
            cif_list[cif_id] = CifPath(cif, graphdata, grid, griddata)

    # get number of split
    if train_fraction + test_fraction > 1:
        raise ValueError(
            f'"train_fraction + test_fraction" must be smaller than 1.0, not {train_fraction + test_fraction}'
        )

    n_total = len(cif_list.keys())
    n_train = int(n_total * train_fraction)
    n_test = int(n_total * test_fraction)
    n_val = n_total - n_train - n_test
    n_split = {"train": n_train, "test": n_test, "val": n_val}

    # remove already-divided values
    for split, direc in split_dir.items():
        split_cifs = {cif.stem for cif in direc.glob("*.cif")}
        for cif in split_cifs:
            if cif in cif_list:
                del cif_list[cif]
                n_split[split] -= 1

    assert sum(n_split.values()) == len(cif_list), "Error! contact with code writer!"
    if not cif_list:  # NO additional divided task
        return

    for split, n in n_split.items():
        if n < -n_total * threshold:
            raise ValueError(
                "{split} folder's cif number is larger than {split}_fraction. change argument {split}_fraction."
            )

    # random split index
    cif_name = sorted(list(cif_list.keys()))
    split_idx = (
        ["train"] * n_split["train"]
        + ["test"] * n_split["test"]
        + ["val"] * n_split["val"]
    )
    np.random.seed(seed=seed)
    np.random.shuffle(split_idx)

    assert len(cif_name) == len(split_idx)

    for cif, split in zip(cif_name, split_idx):
        cifpath = cif_list[cif]
        for suffix in ["cif", "graphdata", "grid", "griddata16"]:
            src = getattr(cifpath, suffix)
            dest = root_dataset / split
            shutil.copy(src, dest)


def _split_json(root_cifs: Path, root_dataset: Path, downstream: str):
    with open(str(root_cifs / f"raw_{downstream}.json")) as f:
        src = json.load(f)
        src = {
            i.replace(".cif", ""): v for i, v in src.items()
        }  # if *.cif in JSON files

    for split in ["train", "test", "val"]:
        cif_folder = root_dataset / split
        cif_list = [cif.stem for cif in cif_folder.glob("*.cif")]
        split_json = {i: src[i] for i in cif_list if i in src}
        with open(str(root_dataset / f"{split}_{downstream}.json"), "w") as f:
            json.dump(split_json, f)


def _make_supercell(atoms, cutoff):
    """
    make atoms into supercell when cell length is less than cufoff (min_length)
    """
    # when the cell lengths are smaller than radius, make supercell to be longer than the radius
    scale_abc = []
    for l in atoms.cell.cellpar()[:3]:
        if l < cutoff:
            scale_abc.append(math.ceil(cutoff / l))
        else:
            scale_abc.append(1)

    # make supercell
    m = np.zeros([3, 3])
    np.fill_diagonal(m, scale_abc)
    atoms = make_supercell(atoms, m)
    return atoms


def make_prepared_data(
    cif: Path, root_dataset_total: Path, logger=None, eg_logger=None, **kwargs
):
    if logger is None:
        logger = get_logger(filename="prepare_data.log")
    if eg_logger is None:
        eg_logger = get_logger(filename="prepare_energy_grid.log")

    if isinstance(cif, str):
        cif = Path(cif)
    if isinstance(root_dataset_total, str):
        root_dataset_total = Path(root_dataset_total)

    root_dataset_total.mkdir(exist_ok=True, parents=True)

    max_length = kwargs.get("max_length", 60.0)
    min_length = kwargs.get("min_length", 30.0)
    max_num_nbr = kwargs.get("max_num_nbr", 12)
    max_num_unique_atoms = kwargs.get("max_num_unique_atoms", 300)
    max_num_atoms = kwargs.get("max_num_atoms", None)
    num_angle_basis = kwargs.get("num_angle_basis", DEFAULT_NUM_ANGLE_BASIS)

    cif_id: str = cif.stem

    p_graphdata = root_dataset_total / f"{cif_id}.graphdata"
    p_griddata = root_dataset_total / f"{cif_id}.griddata16"
    p_grid = root_dataset_total / f"{cif_id}.grid"

    # Grid data and Graph data already exists
    if p_graphdata.exists() and p_griddata.exists() and p_grid.exists():
        logger.info(f"{cif_id} graph data already exists")
        eg_logger.info(f"{cif_id} energy grid already exists")
        return True

    # valid cif check
    try:
        CifParser(cif).get_structures()
    except ValueError as e:
        logger.info(f"{cif_id} failed : {e} (error when reading cif with pymatgen)")
        return False

    # read cif by ASE
    try:
        atoms = read(str(cif))
    except Exception as e:
        logger.error(f"{cif_id} failed : {e}")
        return False

    # 1. get crystal graph
    atoms = _make_supercell(atoms, cutoff=8)  # radius = 8
    if max_num_atoms and len(atoms) > max_num_atoms:
        logger.error(
            f"{cif_id} failed : number of atoms are larger than `max_num_atoms` ({max_num_atoms})"
        )
        return False

    (
        atom_num,
        nbr_idx,
        nbr_dist,
        uni_idx,
        uni_count,
        angle_src,
        angle_dst,
        angle_fea,
    ) = get_crystal_graph(
        atoms, radius=8, max_num_nbr=max_num_nbr, num_angle_basis=num_angle_basis
    )

    if len(nbr_idx) < len(atom_num) * max_num_nbr:
        logger.error(
            f"{cif_id} failed : num_nbr is smaller than max_num_nbr. please make radius larger"
        )
        return False

    if len(uni_idx) > max_num_unique_atoms:
        logger.error(
            f"{cif_id} failed : The number of topologically unique atoms is larget than `max_num_unique_atoms` ({max_num_unique_atoms})"
        )
        return False

    # 2. make supercell with min_length
    atoms_eg = _make_supercell(atoms, cutoff=min_length)
    for l in atoms_eg.cell.cellpar()[:3]:
        if l > max_length:
            logger.error(f"{cif_id} failed : supercell have more than max_length")
            return False

    # 3. calculate energy grid
    eg_success = get_energy_grid(atoms_eg, cif_id, root_dataset_total, eg_logger)

    if eg_success:
        logger.info(f"{cif_id} succeed : supercell length {atoms.cell.cellpar()[:3]}")

        # save cif files
        save_cif_path = root_dataset_total / f"{cif_id}.cif"
        atoms.write(filename=save_cif_path)

        # save graphdata file
        # 新格式：加入 angle_src, angle_dst, angle_fea
        data = [
            cif_id,
            atom_num,
            nbr_idx,
            nbr_dist,
            uni_idx,
            uni_count,
            angle_src,
            angle_dst,
            angle_fea,
        ]
        with open(str(p_graphdata), "wb") as f:
            pickle.dump(data, f)
        return True
    else:
        return False


def _prepare_one_task(task):
    _suppress_noisy_warnings()
    cif, root_dataset_total, kwargs = task
    return make_prepared_data(
        cif=cif,
        root_dataset_total=root_dataset_total,
        logger=None,
        eg_logger=None,
        **kwargs,
    )


def _emit_progress(prefix, done, total, start_time):
    elapsed = max(time.time() - start_time, 1e-9)
    rate = done / elapsed
    remaining = max(total - done, 0)
    eta_seconds = remaining / rate if rate > 0 else float("inf")
    eta_hours = eta_seconds / 3600 if math.isfinite(eta_seconds) else float("inf")
    percent = (done / total * 100.0) if total else 100.0
    print(
        f"[progress] {prefix}: {done}/{total} ({percent:.2f}%) "
        f"elapsed={elapsed/3600:.2f}h rate={rate:.2f} cif/s eta={eta_hours:.2f}h",
        flush=True,
    )


def prepare_data(root_cifs, root_dataset, downstream, **kwargs):
    _suppress_noisy_warnings()
    """
    Args:
        root_cifs (str): root for cif files,
                        it should contains "train" and "test" directory in root_cifs
                        ("val" directory is optional)
        root_dataset (str): root for generated datasets
        downstream (str or list) : name of downstream tasks

    kwargs:
        - seed : (int) random seed for split data. (default : 42)
        - train_fraction : (float) fraction for train dataset. train_fraction + test_fraction must be smaller than 1 (default : 0.8)
        - test_fraction : (float) fraction for test dataset. train_fraction + test_fraction must be smaller than 1 (default : 0.1)

        - get_primitive (bool) : If True, use primitive cell in graph embedding
        - max_num_unique_atoms (int): max number unique atoms in primitive cells (default: 300)
        - max_num_supercell_atoms (int or None): max number atoms in super cell atoms (default: None)
        - max_length (float) : maximum length of supercell
        - min_length (float) : minimum length of supercell
        - max_num_nbr (int) : maximum number of neighbors when calculating graph
        - num_angle_basis (int) : number of RBF bases for angle features (default: 8)
    """
    if not os.path.exists(GRIDAY_PATH):
        raise ImportError(
            "GRIDAY must be installed. \n"
            "Run the following code in bash, \n\n"
            "$ moftransformer install-griday\n\n"
            "or run the following code on Python\n\n"
            ">>> from moftransformer.utils import install_griday\n"
            ">>> install_griday()"
        )

    # set logger
    logger = get_logger(filename="prepare_data.log")
    eg_logger = get_logger(filename="prepare_energy_grid.log")
    print("123")
    # directory to "Path"
    root_cifs = Path(root_cifs)
    root_dataset = Path(root_dataset)

    if not root_cifs.exists():
        raise ValueError(f"{root_cifs} does not exists.")

    # make prepare_data in 'total' directory
    root_dataset_total = Path(root_dataset) / "total"
    root_dataset_total.mkdir(exist_ok=True, parents=True)

    print(f"scanning cif files in {root_cifs} ...")
    cif_paths = sorted(root_cifs.glob("*.cif"))
    print(f"found {len(cif_paths)} cif files, starting preparation")

    num_workers = int(kwargs.get("num_workers", 1))
    chunksize = int(kwargs.get("chunksize", 8))
    progress_every = max(int(kwargs.get("progress_every", 100)), 1)
    print(
        f"using num_workers={num_workers}, chunksize={chunksize}, "
        f"progress_every={progress_every}"
    )

    # make *.grid, *.griddata16, and *.graphdata file
    use_tqdm = sys.stderr.isatty() or sys.stdout.isatty()
    start_time = time.time()
    if num_workers <= 1:
        iterator = cif_paths
        if use_tqdm:
            iterator = tqdm(
                cif_paths,
                total=len(cif_paths),
                desc="Preparing CIFs",
                unit="cif",
            )
        for idx, cif in enumerate(iterator, start=1):
            make_prepared_data(cif, root_dataset_total, logger, eg_logger, **kwargs)
            if (idx % progress_every == 0) or (idx == len(cif_paths)):
                _emit_progress("prepare", idx, len(cif_paths), start_time)
    else:
        tasks = [(cif, root_dataset_total, kwargs) for cif in cif_paths]
        with Pool(processes=num_workers) as pool:
            completed = 0
            progress_bar = None
            if use_tqdm:
                progress_bar = tqdm(
                    total=len(tasks),
                    desc="Preparing CIFs",
                    unit="cif",
                )
            for _ in pool.imap_unordered(
                _prepare_one_task,
                tasks,
                chunksize=chunksize,
            ):
                completed += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                if (completed % progress_every == 0) or (completed == len(tasks)):
                    _emit_progress("prepare", completed, len(tasks), start_time)
            if progress_bar is not None:
                progress_bar.close()

    # automatically split data
    _split_dataset(root_dataset, **kwargs)

    # split json file
    if isinstance(downstream, str):
        _split_json(root_cifs, root_dataset, downstream)
    elif isinstance(downstream, Iterable):
        for single_downstream in downstream:
            _split_json(root_cifs, root_dataset, single_downstream)
    else:
        raise TypeError(f"task must be str or Iterable, not {type(downstream)}")
