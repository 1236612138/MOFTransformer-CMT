import json
import os
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sacred.config.config_summary import ConfigSummary
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, "/home/yihaoyu/docker/bag/MOFTransformer_CMT/MOFTransformer")

import moftransformer
from moftransformer.config import config as _config
from moftransformer.datamodules.datamodule import Datamodule
from moftransformer.modules.module import Module


torch.serialization.add_safe_globals([ConfigSummary])
torch.set_float32_matmul_precision("medium")


def build_config(root_dataset: str, downstream: str):
    config = _config()
    config["root_dataset"] = root_dataset
    config["downstream"] = downstream
    config["test_only"] = True
    config["visualize"] = False
    config["loss_names"] = {
        "ggm": 0,
        "mpp": 0,
        "mtp": 0,
        "vfp": 0,
        "moc": 0,
        "bbc": 0,
        "classification": 0,
        "regression": 1,
    }
    config["mean"] = 0
    config["std"] = 1
    config["batch_size"] = int(os.environ.get("ALIGNN_ENSEMBLE_BATCH_SIZE", "4"))
    config["per_gpu_batchsize"] = 1
    config["img_size"] = int(os.environ.get("ALIGNN_ENSEMBLE_IMG_SIZE", "25"))
    config["max_graph_len"] = int(os.environ.get("ALIGNN_ENSEMBLE_MAX_GRAPH_LEN", "260"))
    config["hid_dim"] = int(os.environ.get("ALIGNN_ENSEMBLE_HID_DIM", "384"))
    config["num_layers"] = int(os.environ.get("ALIGNN_ENSEMBLE_NUM_LAYERS", "8"))
    config["alignn_layers"] = int(os.environ.get("ALIGNN_ENSEMBLE_ALIGNN_LAYERS", "2"))
    config["graph_dropout"] = float(os.environ.get("ALIGNN_ENSEMBLE_GRAPH_DROPOUT", "0.1"))
    config["precision"] = os.environ.get("ALIGNN_ENSEMBLE_PRECISION", "16-mixed")
    config["num_workers"] = int(os.environ.get("ALIGNN_ENSEMBLE_NUM_WORKERS", "4"))
    config["accelerator"] = "gpu" if torch.cuda.is_available() else "cpu"
    config["devices"] = 1
    config["num_nodes"] = 1
    config["strategy"] = None
    return config


def flatten_predict_outputs(rets):
    rows = []
    for ret in rets:
        cif_ids = ret["cif_id"]
        logits = ret["regression_logits"]
        labels = ret["regression_labels"]
        for cif_id, pred, label in zip(cif_ids, logits, labels):
            rows.append((cif_id, float(pred), float(label)))
    return rows


def run_single_predict(ckpt_path: str, config: dict, split: str = "test"):
    model_config = dict(config)
    model_config["load_path"] = ckpt_path
    model = Module(model_config)
    model.eval()
    dm = Datamodule(model_config)
    if split == "test":
        dm.setup("test")
        dataloader = dm.test_dataloader()
    elif split == "val":
        dm.setup("fit")
        dataloader = dm.val_dataloader()
    else:
        raise ValueError(f"Unsupported split: {split}")

    trainer_kwargs = dict(
        accelerator=model_config["accelerator"],
        devices=model_config["devices"],
        num_nodes=model_config["num_nodes"],
        precision=model_config["precision"],
        logger=False,
        enable_checkpointing=False,
        benchmark=True,
        deterministic=True,
        log_every_n_steps=0,
        inference_mode=True,
    )
    strategy = model_config.get("strategy", None)
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy
    trainer = pl.Trainer(**trainer_kwargs)
    rets = trainer.predict(model, dataloader)
    return flatten_predict_outputs(rets)


def average_predictions(prediction_sets):
    by_cif = {}
    for rows in prediction_sets:
        for cif_id, pred, label in rows:
            by_cif.setdefault(cif_id, {"preds": [], "label": label})
            by_cif[cif_id]["preds"].append(pred)
    out = []
    for cif_id, item in sorted(by_cif.items()):
        out.append((cif_id, float(np.mean(item["preds"])), float(item["label"])))
    return out


def weighted_average_predictions(prediction_sets, weights):
    by_cif = {}
    for rows, weight in zip(prediction_sets, weights):
        for cif_id, pred, label in rows:
            by_cif.setdefault(cif_id, {"preds": [], "weights": [], "label": label})
            by_cif[cif_id]["preds"].append(pred)
            by_cif[cif_id]["weights"].append(weight)
    out = []
    for cif_id, item in sorted(by_cif.items()):
        pred = float(np.average(np.array(item["preds"], dtype=np.float32), weights=np.array(item["weights"], dtype=np.float32)))
        out.append((cif_id, pred, float(item["label"])))
    return out


def rows_to_metrics(rows):
    labels = np.array([x[2] for x in rows], dtype=np.float32)
    preds = np.array([x[1] for x in rows], dtype=np.float32)
    return {
        "mse": float(mean_squared_error(labels, preds)),
        "mae": float(mean_absolute_error(labels, preds)),
        "r2_score": float(r2_score(labels, preds)),
    }


def main():
    ckpt_paths = [
        os.environ.get("ALIGNN_ENSEMBLE_CKPT_1", "").strip(),
        os.environ.get("ALIGNN_ENSEMBLE_CKPT_2", "").strip(),
        os.environ.get("ALIGNN_ENSEMBLE_CKPT_3", "").strip(),
    ]
    ckpt_paths = [p for p in ckpt_paths if p]
    if not ckpt_paths:
        raise ValueError("需要提供至少一个 ALIGNN_ENSEMBLE_CKPT_* 路径")

    root_dataset = os.environ.get(
        "ALIGNN_ENSEMBLE_ROOT_DATASET",
        "/home/yihaoyu/docker/bag/MOFTransformer_upgrade/MOFTransformer/all_N2",
    )
    downstream = os.environ.get("ALIGNN_ENSEMBLE_DOWNSTREAM", "N2")
    split = os.environ.get("ALIGNN_ENSEMBLE_SPLIT", "test")
    output_dir = Path(
        os.environ.get(
            "ALIGNN_ENSEMBLE_OUTPUT_DIR",
            "/home/yihaoyu/docker/bag/MOFTransformer_CMT/outputs/all_N2/ensemble_light_chem",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    weighted = os.environ.get("ALIGNN_ENSEMBLE_WEIGHTED", "0").strip() == "1"

    pl.seed_everything(int(os.environ.get("ALIGNN_ENSEMBLE_SEED", "0")))
    config = build_config(root_dataset=root_dataset, downstream=downstream)

    all_preds = []
    val_preds = []
    val_metrics_per_model = []
    for idx, ckpt_path in enumerate(ckpt_paths, start=1):
        print(f"[ensemble] running model {idx}: {ckpt_path}")
        rows = run_single_predict(ckpt_path, config, split=split)
        all_preds.append(rows)

        single_out = output_dir / f"{split}_model{idx}_prediction.csv"
        with single_out.open("w", encoding="utf-8") as f:
            f.write("cif_id,pred,label\n")
            for cif_id, pred, label in rows:
                f.write(f"{cif_id},{pred},{label}\n")

        if weighted:
            val_rows = run_single_predict(ckpt_path, config, split="val")
            val_preds.append(val_rows)
            val_metrics = rows_to_metrics(val_rows)
            val_metrics_per_model.append(val_metrics)
            val_out = output_dir / f"val_model{idx}_prediction.csv"
            with val_out.open("w", encoding="utf-8") as f:
                f.write("cif_id,pred,label\n")
                for cif_id, pred, label in val_rows:
                    f.write(f"{cif_id},{pred},{label}\n")

    if weighted:
        inv_maes = np.array(
            [1.0 / max(m["mae"], 1e-8) for m in val_metrics_per_model],
            dtype=np.float32,
        )
        weights = (inv_maes / inv_maes.sum()).tolist()
        ensemble_rows = weighted_average_predictions(all_preds, weights)
        metrics = rows_to_metrics(ensemble_rows)
        metrics["val_metrics_per_model"] = val_metrics_per_model
        metrics["weights"] = weights
    else:
        ensemble_rows = average_predictions(all_preds)
        metrics = rows_to_metrics(ensemble_rows)

    pred_csv = output_dir / f"{split}_ensemble_prediction.csv"
    with pred_csv.open("w", encoding="utf-8") as f:
        f.write("cif_id,pred,label\n")
        for cif_id, pred, label in ensemble_rows:
            f.write(f"{cif_id},{pred},{label}\n")

    metrics.update({
        "split": split,
        "num_models": len(ckpt_paths),
        "weighted": weighted,
        "ckpt_paths": ckpt_paths,
        "prediction_csv": str(pred_csv),
    })
    metrics_path = output_dir / f"{split}_ensemble_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[ensemble] saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
