import sys
sys.path.insert(0, '/home/yihaoyu/docker/bag/MOFTransformer_upgrade/MOFTransformer')

import inspect
from moftransformer.utils import prepare_data

# 设置路径
root_cifs = "/home/yihaoyu/docker/bag/MOFTransformer_backups/MOFTransformer/all_cif/all.cif"   # 存放 CIF 文件的路径
root_dataset = "/home/yihaoyu/docker/bag/MOFTransformer_upgrade/MOFTransformer/all_N2"  # 处理后数据存放路径
downstream = "N2"   # 目标数据集，包含吸附值

# 训练集和测试集划分比例
train_fraction = 0.8  
test_fraction = 0.1   
print("prepare_data 来自文件：", inspect.getfile(prepare_data))

# 运行数据准备
prepare_data(root_cifs, root_dataset, downstream=downstream, 
             train_fraction=train_fraction, test_fraction=test_fraction)
