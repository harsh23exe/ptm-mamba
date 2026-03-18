import argparse
import json
import os
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

from ptm_classification.datasets import PTMFeaturesDataset
from ptm_classification.metrics import compute_binary_metrics
from ptm_classification.models import CNNGRUClassifier, CNNBiLSTMClassifier


def create_train_val_indices(labels, val_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(int)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size, random_state=seed
    )
    (train_idx, val_idx), = splitter.split(np.zeros_like(labels), labels)
    return train_idx, val_idx


def compute_class_weights(labels) -> torch.Tensor:
    labels = np.asarray(labels).astype(int)
    classes, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    weights = total / (len(classes) * counts.astype(np.float32))
    # ensure 2 classes order [0,1]
    weight_vec = np.ones(2, dtype=np.float32)
    for c, w in zip(classes, weights):
        if 0 <= c < 2:
            weight_vec[c] = w
    return torch.tensor(weight_vec, dtype=torch.float32)


def get_model(
    model_type: str,
    num_classes: int = 2,
    conv_layers: int = 2,
    rnn_layers: int = 1,
) -> nn.Module:
    if model_type == "cnn_gru":
        return CNNGRUClassifier(
            num_classes=num_classes,
            conv_layers=conv_layers,
            rnn_layers=rnn_layers,
        )
    elif model_type == "cnn_bilstm":
        return CNNBiLSTMClassifier(
            num_classes=num_classes,
            conv_layers=conv_layers,
            rnn_layers=rnn_layers,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for x, y, _ in loader:
        x = x.to(device)  # [B, 51, 768]
        y = y.to(device)  # [B]

        optimizer.zero_grad()
        logits = model(x)          # [B, 2]
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(1, total_batches)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_labels = []
    all_probs = []

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        prob = torch.softmax(logits, dim=-1)[:, 1]

        total_loss += loss.item()
        total_batches += 1
        all_labels.append(y.detach().cpu().numpy())
        all_probs.append(prob.detach().cpu().numpy())

    avg_loss = total_loss / max(1, total_batches)
    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    metrics = compute_binary_metrics(y_true, y_prob)
    metrics["loss"] = float(avg_loss)
    return metrics


def run_training(args) -> str:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    train_ds = PTMFeaturesDataset(
        features_root=args.features_root,
        ptm_type=args.ptm_type,
        split="train",
    )

    labels = [train_ds._labels[i] for i in range(len(train_ds))]
    train_idx, val_idx = create_train_val_indices(labels, val_size=args.val_size, seed=args.seed)

    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(train_ds, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    class_weights = compute_class_weights([labels[i] for i in train_idx]).to(device)

    model = get_model(
        args.model_type,
        num_classes=2,
        conv_layers=args.conv_layers,
        rnn_layers=args.rnn_layers,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_root, args.ptm_type, args.model_type, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    config = vars(args).copy()
    config["class_weights"] = class_weights.detach().cpu().tolist()
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    best_metric = -float("inf")
    best_state_path = os.path.join(run_dir, "model_best.pt")

    history = []
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
        }
        history.append(record)

        current = val_metrics.get("mcc", 0.0)
        if current > best_metric:
            best_metric = current
            torch.save(model.state_dict(), best_state_path)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_mcc={val_metrics['mcc']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

    with open(os.path.join(run_dir, "metrics_val.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Best model saved to: {best_state_path}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train PTM classifier on fixed PTM-Mamba embeddings."
    )
    parser.add_argument("--ptm_type", default="acet_k")
    parser.add_argument("--features_root", default="features")
    parser.add_argument("--output_root", default="classifier_runs")
    parser.add_argument("--model_type", choices=["cnn_gru", "cnn_bilstm"], default="cnn_gru")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conv_layers", type=int, default=2)
    parser.add_argument("--rnn_layers", type=int, default=1)
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()

