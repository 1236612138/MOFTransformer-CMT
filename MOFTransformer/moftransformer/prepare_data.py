import os
import importlib

# 1. 数据位置
root_cifs = "/home/yihaoyu/dataset/pcd_data/data.cif/all"
root_dataset = "/home/yihaoyu/YC/MOFTransformer/MOFTransformer/cofdata"
downstream_name = "N2"

# 步骤 A: 导入 prepare_data.py 所在的整个模块
prep_data_module = importlib.import_module("moftransformer.utils.prepare_data")

# 步骤 B: 手动覆盖模块里的全局变量 GRIDAY_PATH
prep_data_module.GRIDAY_PATH = os.path.join(
    "/home/yihaoyu/anaconda3/envs/trans/lib/python3.12/site-packages/moftransformer/libs/GRIDAY/scripts/grid_gen"
)

# 步骤 C: 调用模块中的 prepare_data() 函数
prep_data_module.prepare_data(
    root_cifs=root_cifs,
    root_dataset=root_dataset,
    downstream=downstream_name,
    train_fraction=0.8,
    test_fraction=0.1,
    seed=42
)  