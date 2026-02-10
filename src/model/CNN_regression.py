import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
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
    image_size: int = 224
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    log_every_n_batches: int = 4

    base_path: str = "/kaggle/input/depthmapdbh-da/DepthMapDBH2023/"
    segmentation_model_name: str = "DA3_LARGE"
    models = [
        "resnet50"
        # "vit_base_patch32_clip_448"
    ]


def get_train_transforms(image_size, in_channels=3):
    transforms = [
        T.Resize((image_size, image_size)),
        # T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
    ]

    if in_channels == 1:
        # replicate 1 → 3 channels for ImageNet backbones
        transforms.append(T.Lambda(lambda x: x.repeat(3, 1, 1)))

    transforms.append(
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    )

    return T.Compose(transforms)


def get_eval_transforms(image_size, in_channels=3):
    transforms = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ]

    if in_channels == 1:
        transforms.append(T.Lambda(lambda x: x.repeat(3, 1, 1)))

    transforms.append(
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    )

    return T.Compose(transforms)


def filter_dbh_dataframe(
    df: pd.DataFrame,
    min_dbh: float = 0.0,
    verbose: bool = True,
) -> pd.DataFrame:
    if "DBH" not in df.columns:
        raise ValueError("DataFrame must contain 'DBH' column")

    before = len(df)
    df = df[df["DBH"] > min_dbh].copy()
    df.reset_index(drop=True, inplace=True)
    after = len(df)

    if verbose:
        print(f"[DBH Filter] DBH > {min_dbh}: {before} → {after}")

    return df


class DBHDepthDataset(Dataset):
    def __init__(self, csv_file, base_path, transform=None):
        self.base_path = Path(base_path)
        df = pd.read_csv(csv_file)
        self.data = filter_dbh_dataframe(df, min_dbh=0)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        depth_map_path = self.base_path / Path(row["depth_anything_maps_path"].replace("\\", "/"))
        x = np.load(depth_map_path).astype(np.float32)
        x = np.clip(x, 0, 255).astype(np.uint8)
        x = Image.fromarray(x, mode="L")
        if self.transform is not None:
            x = self.transform(x)

        y = row["DBH"]
        y = np.log1p(y)

        y = torch.tensor(y, dtype=torch.float32)

        return x, y


class DBHImageDataset(Dataset):
    def __init__(self, csv_file, base_path, transform=None):
        self.base_path = Path(base_path)
        df = pd.read_csv(csv_file)
        self.data = filter_dbh_dataframe(df, min_dbh=0)

        self.transform = transform

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = self.base_path / row["image_path"].split("/")[0] / row["image_path"]
        img = Image.open(img_path).convert("RGB")

        x = self.transform(img) if self.transform else T.ToTensor()(img)

        y = row["DBH"]
        y = np.log1p(y)

        y = torch.tensor(y, dtype=torch.float32)
        return x, y


class DBHRegressor(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()

        # self.input_conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,  # no classifier
            global_pool="avg",
        )

        self.head = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        # x = self.input_conv(x)
        x = self.backbone(x)
        return self.head(x).squeeze(1)


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    epoch,
    log_every_n_batches=None,
):
    model.train()
    total_loss = 0.0

    for batch_idx, (x, y) in enumerate(tqdm(loader, desc="Train", leave=False)):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y)

        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * x.size(0)

        if log_every_n_batches and batch_idx % log_every_n_batches == 0:
            wandb.log(
                {
                    "train/batch_loss": batch_loss,
                    "epoch": epoch,
                },
                step=epoch * len(loader) + batch_idx,
            )

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

    train_tfms = get_train_transforms(cfg.image_size, in_channels=1)
    eval_tfms = get_eval_transforms(cfg.image_size, in_channels=1)

    full_dataset = DBHDepthDataset(
        train_csv,
        cfg.base_path,
        transform=train_tfms,
    )

    test_dataset = DBHDepthDataset(
        test_csv,
        cfg.base_path,
        transform=eval_tfms,
    )

    # full_dataset = DBHImageDataset(
    #     train_csv,
    #     cfg.base_path,
    #     transform=train_tfms,
    # )

    # test_dataset = DBHImageDataset(
    #     test_csv,
    #     cfg.base_path,
    #     transform=eval_tfms,
    # )

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

        run = wandb.init(
            project="DBH-Depth-Map-CNN-Regression",
            name=run_name,
            config=vars(cfg),
        )

        model = DBHRegressor(backbone).to(cfg.device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
        loss_fn = nn.MSELoss()

        metrics = {
            "rmse": MeanSquaredError(squared=False).to(cfg.device),
            "mae": MeanAbsoluteError().to(cfg.device),
            "r2": R2Score().to(cfg.device),
        }

        best_val = float("inf")
        patience, wait = 26, 0

        for epoch in tqdm(range(cfg.epochs)):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, loss_fn, cfg.device, epoch, cfg.log_every_n_batches
            )

            val_metrics = evaluate(model, val_loader, loss_fn, metrics, cfg.device)

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_metrics["loss"],
                    "val/rmse": val_metrics["rmse"],
                    "val/mae": val_metrics["mae"],
                    "val/r2": val_metrics["r2"],
                }
            )

            print(f"[{epoch:03d}] train={train_loss:.4f} val_rmse={val_metrics['rmse']:.3f}")

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                wait = 0

                ckpt_path = f"{run_name}_{epoch}_best.pt"
                torch.save(model.state_dict(), ckpt_path)

                print(
                    f"[LOG] Epoch {epoch:03d} | "
                    f"New best val loss: {best_val:.6f} | "
                    f"Checkpoint saved → {ckpt_path}"
                )

                artifact = wandb.Artifact(
                    name=f"{backbone}-dbh-regressor",
                    type="model",
                    metadata={
                        "backbone": backbone,
                        "segmentation_model": cfg.segmentation_model_name,
                        "epoch": epoch,
                        "val_loss": best_val,
                    },
                )

                artifact.add_file(ckpt_path, overwrite=True)
                run.log_artifact(artifact)

        # Test
        model.load_state_dict(torch.load(f"{run_name}_{epoch}_best.pt"))
        test_metrics = evaluate(model, test_loader, loss_fn, metrics, cfg.device)

        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        wandb.finish()

    print("All trainings done!")


if __name__ == "__main__":
    main()
