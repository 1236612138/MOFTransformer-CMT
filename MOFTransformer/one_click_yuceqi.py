import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sacred.config.config_summary import ConfigSummary
import torch.serialization
torch.serialization.add_safe_globals([ConfigSummary])

sys.path.insert(0, '/home/yihaoyu/docker/MOFTransformer/MOFTransformer')

from moftransformer.utils import prepare_data
from moftransformer.config import config as get_default_config
from moftransformer.datamodules.datamodule import Datamodule
from moftransformer.modules.module import Module

# ========== 目录配置 ==========
root_cif_input = "./yuceqitest"  # 放置你要预测的 .cif 文件
predict_dataset_dir = "./yuceqitest_dataset1"  # 输出预测用数据集
output_csv = os.path.join(predict_dataset_dir, "predictions.csv")

model_ckpt_path = "/home/yihaoyu/docker/MOFTransformer/MOFTransformer/haoyu_cif_N2/pretrained_mof_seed0_from_pmtransformer/version_3/checkpoints/best.ckpt"
mean, std = 0.0, 1.0

def reverse_normalization(pred, mean, std):
    return pred * std + mean

def generate_test_json(test_dir, json_path):
    entries = [f for f in os.listdir(test_dir) if f.endswith(".cif")]
    cof_ids = {os.path.splitext(f)[0]: 0 for f in entries}
    with open(json_path, "w") as f:
        json.dump(cof_ids, f, indent=4)
    print(f"✅ test_N2.json saved to {json_path}")

def generate_raw_json(cif_dir, json_path):
    entries = [f for f in os.listdir(cif_dir) if f.endswith(".cif")]
    dummy = {os.path.splitext(f)[0]: 0 for f in entries}
    with open(json_path, "w") as f:
        json.dump(dummy, f, indent=4)
    print(f"✅ raw_N2.json saved to {json_path}")

def main():
    # Step 0: 自动生成 raw_N2.json
    raw_json_path = os.path.join(root_cif_input, "raw_N2.json")
    if not os.path.exists(raw_json_path):
        generate_raw_json(root_cif_input, raw_json_path)

    print("🛠 Step 1: prepare_data")
    prepare_data(
        root_cifs=root_cif_input,
        root_dataset=predict_dataset_dir,
        downstream="N2",
        train_fraction=0.0,
        test_fraction=1.0
    )

    # Step 2: 自动生成 test_N2.json
    test_dir = os.path.join(predict_dataset_dir, "test")
    json_path = os.path.join(predict_dataset_dir, "test_N2.json")
    if not os.path.exists(json_path):
        generate_test_json(test_dir, json_path)

    print("🧠 Step 3: 初始化配置")
    config = get_default_config()
    config.update({
        "root_dataset": predict_dataset_dir,
        "downstream": "N2",
        "num_workers": 0,
        "per_gpu_batchsize": 8,
        "draw_false_grid": False,
        "img_size": 30,
        "patch_size": 5,
        "nbr_fea_len": 64,
        "hid_dim": 768,
        "loss_names": {
            "ggm": 0, "mpp": 0, "mtp": 0,
            "vfp": 0, "bbc": 0, "moc": 0,
            "classification": 0, "regression": 1
        },
        "mean": mean,
        "std": std
    })

    # Step 4: 加载数据与模型
    datamodule = Datamodule(config)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    model = Module.load_from_checkpoint(model_ckpt_path, config=config)
    model.eval()
    model.current_tasks = ["regression"]

    predictions = []
    names = []

    print("🔮 Step 5: 开始预测")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            output = model(batch)
            preds = output["regression_logits"]
            if isinstance(preds, torch.Tensor):
                preds = preds.cpu().tolist()
            predictions.extend(preds)
            names.extend(output["cif_id"])

    predictions = np.array(predictions)
    predictions = reverse_normalization(predictions, mean, std)

    df = pd.DataFrame({
        "COF_name": names,
        "Predicted_N2_Adsorption": predictions
    })
    df.to_csv(output_csv, index=False)
    print(f"✅ Prediction saved to: {output_csv}")

if __name__ == "__main__":
    main()
