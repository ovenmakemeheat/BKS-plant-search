import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from tqdm.auto import tqdm

load_dotenv()


@dataclass
class Config:
    epochs: int = 300
    batch_size: int = 32
    lr: float = 1e-3
    image_size: int = 192
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    base_path: str = "../dataset/DepthMapDBH2023/"
    segmentation_model_name: str = "DA3_LARGE"
    models = [
        "mobilenetv2_100",
        "mobilenetv3_small_100",
        "densenet121",
        "efficientnet_b0",
    ]


class DBHDepthDataset(Dataset):
    def __init__(self, csv_file, base_path):
        self.base_path = Path(base_path)
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load depth map from .npy file (504, 504)
        depth_map_path = self.base_path / row["depth_anything_maps_path"]
        x = np.load(depth_map_path).astype(np.float32)

        # Normalize to [0, 1] range
        x = x / 255.0

        # Add channel dimension: [H, W] → [1, H, W]
        x = torch.from_numpy(x).unsqueeze(0)

        y = torch.tensor(row["DBH"], dtype=torch.float32)

        return x, y


class DBHRegressor(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()

        self.input_conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,  # no classifier
            global_pool="avg",
        )

        self.head = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.backbone(x)
        return self.head(x).squeeze(1)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)

    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y)

        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * x.size(0)

        pbar.set_postfix(loss=batch_loss)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, metrics, device):
    model.eval()
    total_loss = 0.0

    for m in metrics.values():
        m.reset()

    pbar = tqdm(loader, desc="Eval", leave=False)

    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        preds = model(x)

        batch_loss = loss_fn(preds, y).item()
        total_loss += batch_loss * x.size(0)

        for m in metrics.values():
            m.update(preds, y)

        pbar.set_postfix(loss=batch_loss)

    results = {name: m.compute().item() for name, m in metrics.items()}
    results["loss"] = total_loss / len(loader.dataset)
    return results


def main():
    cfg = Config()
    seed = random.randint(0, 10_000)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load datasets using CSV files
    train_csv = Path(cfg.base_path) / "train/train/files_with_depth_maps_DA3_LARGE.csv"
    test_csv = Path(cfg.base_path) / "test/test/files_with_depth_maps_DA3_LARGE.csv"

    full_dataset = DBHDepthDataset(train_csv, cfg.base_path)
    test_dataset = DBHDepthDataset(test_csv, cfg.base_path)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    for backbone in cfg.models:
        run_name = (
            f"{backbone}_depth_dbh_{cfg.segmentation_model_name}_{datetime.now():%Y%m%d_%H%M}"
        )

        wandb.init(
            project="DBH-Depth-Map-CNN-Regression",
            name=run_name,
            config=vars(cfg),
        )

        model = DBHRegressor(backbone).to(cfg.device)

        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
        loss_fn = nn.MSELoss()

        metrics = {
            "rmse": MeanSquaredError(squared=False).to(cfg.device),
            "mae": MeanAbsoluteError().to(cfg.device),
            "r2": R2Score().to(cfg.device),
        }

        best_val = float("inf")
        patience, wait = 26, 0

        for epoch in range(cfg.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, cfg.device)

            val_metrics = evaluate(model, val_loader, loss_fn, metrics, cfg.device)

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            )

            print(f"[{epoch:03d}] train={train_loss:.4f} val_rmse={val_metrics['rmse']:.3f}")

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                wait = 0
                torch.save(
                    model.state_dict(),
                    f"{run_name}_best.pt",
                )
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping")
                    break

        # Test
        model.load_state_dict(torch.load(f"{run_name}_best.pt"))
        test_metrics = evaluate(model, test_loader, loss_fn, metrics, cfg.device)

        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        wandb.finish()

    print("All trainings done!")


if __name__ == "__main__":
    main()
