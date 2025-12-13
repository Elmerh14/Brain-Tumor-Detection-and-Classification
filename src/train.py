# src/train.py
from pathlib import Path
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import FolderImageDataset, find_data_root
from model import build_resnet50


def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def evaluate(model, loader, loss_fn, device, desc="Eval"):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item()
            total_acc += accuracy(logits, y)

    total_loss /= max(1, len(loader))
    total_acc /= max(1, len(loader))
    return total_loss, total_acc


def run(use_augmentation: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Experiment knobs
    # -------------------------
    img_size = 224
    batch_size = 32
    lr = 1e-4
    epochs = 15
    val_ratio = 0.2
    seed = 42

    data_root = find_data_root()
    print("Using data root:", data_root)
    print("Device:", device)
    print("Augmentation:", "ON" if use_augmentation else "OFF")

    # -------------------------
    # Build datasets
    # -------------------------
    # Train dataset: augmentation depends on the experiment
    train_full = FolderImageDataset(
        data_root, split="Training", img_size=img_size, train=use_augmentation
    )
    # Val dataset: NEVER augmented (deterministic)
    val_full = FolderImageDataset(
        data_root, split="Training", img_size=img_size, train=False
    )

    # Deterministic split of indices (same split for both datasets)
    n = len(train_full)
    val_size = int(n * val_ratio)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    # Test set: untouched and deterministic
    test_ds = FolderImageDataset(
        data_root, split="Testing", img_size=img_size, train=False
    )

    # -------------------------
    # Loaders (Windows-friendly)
    # -------------------------
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # -------------------------
    # Model / Optim
    # -------------------------
    model = build_resnet50(num_classes=4, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Always save to project-root /runs regardless of where you run from
    out_dir = Path(__file__).resolve().parent.parent / "runs"

    out_dir.mkdir(exist_ok=True)

    best_path = out_dir / ("best_aug.pt" if use_augmentation else "best_noaug.pt")
    history_path = out_dir / ("history_aug.csv" if use_augmentation else "history_noaug.csv")

    print("Will save model to:", best_path.resolve())
    print("Will save history to:", history_path.resolve())


    # init history CSV
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_acc = 0.0

    # -------------------------
    # Train
    # -------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs} [Train]", leave=True):
            x, y = x.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            train_loss += loss.item()
            train_acc += accuracy(logits.detach(), y)

        train_loss /= max(1, len(train_loader))
        train_acc  /= max(1, len(train_loader))

        val_loss, val_acc = evaluate(
            model, val_loader, loss_fn, device,
            desc=f"Epoch {epoch:02d}/{epochs} [Val]"
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        # log history row
        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        # save best by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print("Best val acc:", best_val_acc)
    print("Saved best model:", best_path)
    print("Saved history:", history_path)

    # -------------------------
    # Final Test (once)
    # -------------------------
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device, desc="Final [Test]")
    print(f"FINAL TEST | loss {test_loss:.4f} acc {test_acc:.4f}")


if __name__ == "__main__":
    run()