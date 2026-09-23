import json
import numpy as np

# ==== 修改成你的 train_N2.json 路径 ====
train_json = "/home/yihaoyu/docker/bag/MOFTransformer_upgrade/MOFTransformer/all_N2/train_N2.json"

with open(train_json, "r") as f:
    data = json.load(f)

values = np.array(list(data.values()), dtype=float)

mean = float(values.mean())
std = float(values.std())

print(f"Train N2 mean: {mean:.6f}")
print(f"Train N2 std : {std:.6f}")
