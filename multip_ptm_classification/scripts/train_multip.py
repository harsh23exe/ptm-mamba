import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from contextlib import nullcontext

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import csv as csv_mod

from multip_ptm_classification.datasets import MultiPTMFeaturesDataset
from multip_ptm_classification.metrics import compute_binary_metrics, metrics_per_ptm, aggregate_macro
from multip_ptm_classification.models import PTMConditionedCNNGRU
from ptm_classification.models import CNNGRUClassifier, CNNBiLSTMClassifier


def create_train_val_indices(labels, val_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(int)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    (train_idx, val_idx), = splitter.split(np.zeros_like(labels), labels)
    return train_idx, val_idx


def compute_class_weights(labels) -> torch.Tensor:
    labels = np.asarray(labels).astype(int)
    classes, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    weights = total / (len(classes) * counts.astype(np.float32))
    weight_vec = np.ones(2, dtype=np.float32)
    for c, w in zip(classes, weights):
        if 0 <= c < 2:
            weight_vec[c] = w
    return torch.tensor(weight_vec, dtype=torch.float32)


def _balanced_ptm_sampler(ptm_ids: np.ndarray) -> WeightedRandomSampler:
    ptm_ids = np.asarray(ptm_ids).astype(int)
    counts = np.bincount(ptm_ids, minlength=ptm_ids.max() + 1).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts[ptm_ids]
    w = torch.from_numpy(inv.astype(np.float32))
    return WeightedRandomSampler(w, num_samples=len(w), replacement=True)


def get_model(args, num_ptm: int) -> nn.Module:
    if args.model_type == "cnn_gru":
        if args.condition_on_ptm:
            return PTMConditionedCNNGRU(
                num_ptm_types=num_ptm,
                ptm_embed_dim=args.ptm_embed_dim,
                conv_layers=args.conv_layers,
                rnn_layers=args.rnn_layers,
                num_classes=2,
            )
        return CNNGRUClassifier(
            num_classes=2,
            conv_layers=args.conv_layers,
            rnn_layers=args.rnn_layers,
        )
    if args.model_type == "cnn_bilstm":
        if args.condition_on_ptm:
            raise ValueError(
                "condition_on_ptm is only implemented for cnn_gru in this module."
            )
        return CNNBiLSTMClassifier(
            num_classes=2,
            conv_layers=args.conv_layers,
            rnn_layers=args.rnn_layers,
        )
    raise ValueError(f"Unknown model_type: {args.model_type}")


def _autocast_context(device: torch.device, use_amp: bool):
    if not use_amp or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    condition_on_ptm: bool,
    use_amp: bool,
    scaler,
    grad_accum_steps: int,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        x, y, meta = batch
        if condition_on_ptm:
            ptm_ids = torch.tensor([m["ptm_id"] for m in meta], dtype=torch.long, device=device)
        else:
            ptm_ids = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with _autocast_context(device, use_amp):
            if condition_on_ptm:
                logits = model(x, ptm_ids)
            else:
                logits = model(x)
            loss = criterion(logits, y)
            loss_for_backward = loss / max(1, grad_accum_steps)

        scaler.scale(loss_for_backward).backward()

        do_step = ((step + 1) % max(1, grad_accum_steps) == 0) or (step + 1 == len(loader))
        if do_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        n += 1
    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(model, loader, criterion, device, condition_on_ptm: bool, use_amp: bool):
    model.eval()
    total_loss = 0.0
    n = 0
    all_labels = []
    all_probs = []
    for batch in loader:
        x, y, meta = batch
        if condition_on_ptm:
            ptm_ids = torch.tensor([m["ptm_id"] for m in meta], dtype=torch.long, device=device)
        else:
            ptm_ids = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with _autocast_context(device, use_amp):
            if condition_on_ptm:
                logits = model(x, ptm_ids)
            else:
                logits = model(x)
            loss = criterion(logits, y)
        prob = torch.softmax(logits, dim=-1)[:, 1]
        total_loss += loss.item()
        n += 1
        all_labels.append(y.detach().cpu().numpy())
        all_probs.append(prob.detach().cpu().numpy())
    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    m = compute_binary_metrics(y_true, y_prob)
    m["loss"] = float(total_loss / max(1, n))
    return m


def collate_with_meta(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    meta = [b[2] for b in batch]
    return xs, ys, meta


def parse_ptm_list(s: str) -> List[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def run_training(args) -> str:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    ptm_types = parse_ptm_list(args.ptm_types)

    full_train = MultiPTMFeaturesDataset(
        features_root=args.features_root,
        ptm_types=ptm_types,
        split="train",
    )
    labels = [full_train._labels[i] for i in range(len(full_train))]
    ptm_ids_all = np.array([full_train._ptm_ids[i] for i in range(len(full_train))], dtype=int)

    train_idx, val_idx = create_train_val_indices(labels, val_size=args.val_size, seed=args.seed)
    train_subset = Subset(full_train, train_idx)
    val_subset = Subset(full_train, val_idx)

    condition = args.condition_on_ptm and args.model_type == "cnn_gru"
    collate_fn = collate_with_meta

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=not args.balance_ptm_batches,
        sampler=_balanced_ptm_sampler(ptm_ids_all[train_idx]) if args.balance_ptm_batches else None,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    class_weights = compute_class_weights([labels[i] for i in train_idx]).to(device)
    model = get_model(args, num_ptm=len(ptm_types)).to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_root, "all_ptms_model", args.model_type, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    cfg = vars(args).copy()
    cfg["ptm_types_resolved"] = ptm_types
    cfg["class_weights"] = class_weights.detach().cpu().tolist()
    cfg["num_cuda_visible"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    best = -float("inf")
    best_path = os.path.join(run_dir, "model_best.pt")
    history = []

    for epoch in range(1, args.num_epochs + 1):
        tr = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            condition,
            args.amp,
            scaler,
            args.grad_accum_steps,
        )
        vm = evaluate(model, val_loader, criterion, device, condition, args.amp)
        history.append({"epoch": epoch, "train_loss": tr, "val": vm})
        cur = vm.get("mcc", 0.0)
        if cur > best:
            best = cur
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, best_path)
        print(
            f"Epoch {epoch:03d} | train_loss={tr:.4f} | val_loss={vm['loss']:.4f} | "
            f"val_mcc={vm['mcc']:.4f} | val_f1={vm['f1']:.4f} | val_aupr={vm.get('aupr', float('nan')):.4f}"
        )

    with open(os.path.join(run_dir, "metrics_val.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best model saved to: {best_path}")

    # ── Auto-evaluate on test set per PTM ────────────────────────────────────
    print("\n========== Test evaluation (per PTM) ==========")
    test_ds = MultiPTMFeaturesDataset(
        features_root=args.features_root,
        ptm_types=ptm_types,
        split="test",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    eval_model = get_model(args, num_ptm=len(ptm_types)).to(device)
    eval_model.load_state_dict(torch.load(best_path, map_location=device))

    id_to_type = test_ds.ptm_id_to_type()
    ptm_names = [id_to_type[i] for i in range(len(ptm_types))]

    eval_model.eval()
    all_labels, all_probs, all_pids = [], [], []
    with torch.no_grad():
        for x, y, meta in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with _autocast_context(device, args.amp):
                if condition:
                    pid_t = torch.tensor([m["ptm_id"] for m in meta], dtype=torch.long, device=device)
                    logits = eval_model(x, pid_t)
                else:
                    logits = eval_model(x)
            prob = torch.softmax(logits, dim=-1)[:, 1]
            all_labels.append(y.cpu().numpy())
            all_probs.append(prob.cpu().numpy())
            all_pids.append(np.array([m["ptm_id"] for m in meta], dtype=int))

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    ptm_id_arr = np.concatenate(all_pids)

    overall = compute_binary_metrics(y_true, y_prob)
    per_ptm = metrics_per_ptm(y_true, y_prob, ptm_id_arr, ptm_names=ptm_names)
    macro = aggregate_macro(per_ptm, ["mcc", "f1", "aupr", "auroc"])

    # Write CSV
    csv_rows = []
    for name in ptm_names:
        m = per_ptm.get(name, {})
        csv_rows.append({
            "ptm": name,
            "model_checkpoint": "model_best.pt",
            "split": "test",
            "mcc": m.get("mcc", ""),
            "f1": m.get("f1", ""),
            "auroc": m.get("auroc", ""),
            "aupr": m.get("aupr", ""),
            "precision": m.get("precision", ""),
            "recall": m.get("recall", ""),
            "n_rows": int(m.get("n", 0)),
        })

    csv_path = os.path.join(run_dir, "metrics_test_per_ptm.csv")
    fieldnames = ["ptm", "model_checkpoint", "split", "mcc", "f1", "auroc", "aupr", "precision", "recall", "n_rows"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    # Write JSON
    json_out = {
        "split": "test",
        "model_checkpoint": os.path.abspath(best_path),
        "overall": overall,
        "per_ptm": per_ptm,
        "macro": macro,
        "ptm_types": ptm_types,
    }
    json_path = os.path.join(run_dir, "metrics_test_per_ptm.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)

    # Print table
    header = f"{'ptm':<12} {'mcc':>8} {'f1':>8} {'auroc':>8} {'aupr':>8} {'precision':>10} {'recall':>8} {'n_rows':>8}"
    print(header)
    print("-" * len(header))
    for row in csv_rows:
        def fmt(v):
            if v == "" or v is None:
                return "     N/A"
            return f"{float(v):8.4f}"
        print(
            f"{row['ptm']:<12} {fmt(row['mcc'])} {fmt(row['f1'])} "
            f"{fmt(row['auroc'])} {fmt(row['aupr'])} {fmt(row['precision']):>10} "
            f"{fmt(row['recall'])} {row['n_rows']:>8}"
        )
    print()
    print(f"Macro avg  | MCC={macro.get('macro_mcc',0):.4f} | F1={macro.get('macro_f1',0):.4f} "
          f"| AUROC={macro.get('macro_auroc',0):.4f} | AUPR={macro.get('macro_aupr',0):.4f}")
    print(f"\nSaved CSV : {csv_path}")
    print(f"Saved JSON: {json_path}")

    return run_dir


def main():
    p = argparse.ArgumentParser(description="Train one classifier on pooled multi-PTM features.")
    p.add_argument(
        "--ptm_types",
        required=True,
        help="Comma-separated folder names under features_root, e.g. acet_k,phos_s,phos_y,phos_t",
    )
    p.add_argument("--features_root", default="features")
    p.add_argument("--output_root", default="classifier_runs")
    p.add_argument("--model_type", choices=["cnn_gru", "cnn_bilstm"], default="cnn_gru")
    p.add_argument("--condition_on_ptm", action="store_true")
    p.add_argument("--ptm_embed_dim", type=int, default=32)
    p.add_argument("--balance_ptm_batches", action="store_true", help="Sample equally often per PTM id.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--conv_layers", type=int, default=2)
    p.add_argument("--rnn_layers", type=int, default=1)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision (recommended on A100).")
    p.add_argument("--data_parallel", action="store_true", help="Use torch.nn.DataParallel across visible GPUs.")
    p.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps.")
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader worker count.")
    p.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader prefetch factor.")
    args = p.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
