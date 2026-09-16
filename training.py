from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import os


import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.tensorboard import SummaryWriter

from utils import seed_everything, free_gpu
from dataset import create_cases_dataframe, create_train_val_indices
from metrics import DaTClassifierMetrics

import logging 
# ======== # CONFIGURATION ======

logging.basicConfig( level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", ) 
logger = logging.getLogger(__name__)





SEED = 26
seed_everything(SEED)


DATA_ROOT = Path("./data/")
NIFTI_DIR = DATA_ROOT / "niftis"
CSV_PATH = DATA_ROOT / "train_labels.csv"

CKPTS_DIR = Path("./ckpts/")
CKPTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Loading labels from: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)




"""# Training pipeline"""

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, CropForegroundd, ResizeWithPadOrCropd, EnsureTyped, NormalizeIntensityd
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import DenseNet121

TARGET_SIZE = (112, 112, 80)
TARGET_SPACING = (2.46, 2.46, 2.46)
BATCH_SIZE = 4


train_transforms = Compose([
    LoadImaged(keys=["image"], image_only=True),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=TARGET_SPACING, mode="bilinear"),
    CropForegroundd(keys=["image"], source_key="image"),
    NormalizeIntensityd(keys=["image"], nonzero=True),
    ResizeWithPadOrCropd(keys=["image"], spatial_size=TARGET_SIZE),
    EnsureTyped(keys=["image", "label"])
])

val_transforms = Compose([
    LoadImaged(keys=["image"], image_only=True),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=TARGET_SPACING, mode="bilinear"),
    CropForegroundd(keys=["image"], source_key="image"),
    NormalizeIntensityd(keys=["image"], nonzero=True),
    ResizeWithPadOrCropd(keys=["image"], spatial_size=TARGET_SIZE),
    EnsureTyped(keys=["image", "label"])
])

logger.info("Creating stratified train/validation split...")

cases = create_cases_dataframe(df, NIFTI_DIR)
labels = np.array([c["label"] for c in cases])
indices = np.arange(len(cases))

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)

train_idx, val_idx = next(splitter.split(indices, labels))
train_cases = [cases[i] for i in train_idx]
val_cases = [cases[i] for i in val_idx]


train_dataset = CacheDataset(data=train_cases, transform=train_transforms, cache_rate=0.5, num_workers=5)
val_dataset = CacheDataset(data=val_cases, transform=val_transforms, cache_rate=0.5, num_workers=5)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

logger.info("Datasets ready")



model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

#scaler = torch.cuda.amp.GradScaler()



EPOCHS = 40
best_val_loss = np.inf
patience = 10
counter = 0

MODEL_PATH = CKPTS_DIR / "best_DATScanModel.pth"
writer = SummaryWriter(log_dir="./logs/")

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # ======================
    # TRAIN
    # ======================

    model.train()
    train_losses = []

    for batch in tqdm(train_loader):

        images = batch["image"].to(device)
        labels = batch["label"].to(device).float().view(-1, 1)

        optimizer.zero_grad()
        #with torch.cuda.amp.autocast():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_losses.append(loss.item())

    train_loss = np.mean(train_losses)

    # ======================
    # VALIDATION
    # ======================

    model.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():

        for batch in tqdm(val_loader):

            images = batch["image"].to(device)
            labels = batch["label"].to(device).float().view(-1, 1)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            y_true.extend(labels.cpu().numpy().flatten())
            y_prob.extend(probs.cpu().numpy().flatten())

    val_metrics = DaTClassifierMetrics(y_true, y_prob)

    val_loss = val_metrics["log_loss"]
    val_auc = val_metrics["roc_auc"]

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/validation", val_loss, epoch)
    writer.add_scalar("Metrics/validation", val_auc, epoch)

    print(f"Train loss : {train_loss:.4f}")
    print(f"Val logloss: {val_loss:.4f}")
    print(f"Val AUROC  : {val_auc:.4f}")

    # ======================
    # SAVE BEST MODEL
    # ======================

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        counter = 0

        torch.save(model.state_dict(), MODEL_PATH)

        print("Saved new best model")

    else:

        counter += 1

        print(f"Patience {counter}/{patience}")

        if counter >= patience:

            print("Early stopping")
            break