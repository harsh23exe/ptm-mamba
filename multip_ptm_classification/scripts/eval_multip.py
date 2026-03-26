import argparse
import csv
import json
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from multip_ptm_classification.datasets import MultiPTMFeaturesDataset
from multip_ptm_classification.metrics import (
    aggregate_macro,
    compute_binary_metrics,
    metrics_per_ptm,
)
from multip_ptm_classification.models import PTMConditionedCNNGRU
from ptm_classification.models import CNNGRUClassifier, CNNBiLSTMClassifier


def parse_ptm_list(s: str) -> List[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def collate(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    meta = [b[2] for b in batch]
    return xs, ys, meta


def build_model(args, num_ptm: int) -> nn.Module:
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
            raise ValueError("condition_on_ptm only supported for cnn_gru.")
        return CNNBiLSTMClassifier(
            num_classes=2,
            conv_layers=args.conv_layers,
            rnn_layers=args.rnn_layers,
        )
    raise ValueError(f"Unknown model_type: {args.model_type}")


@torch.no_grad()
def evaluate_full(model, loader, device, condition_on_ptm: bool):
    model.eval()
    all_labels = []
    all_probs = []
    all_ptm_ids = []
    for x, y, meta in loader:
        x = x.to(device)
        y = y.to(device)
        if condition_on_ptm:
            ptm_ids = torch.tensor([m["ptm_id"] for m in meta], dtype=torch.long, device=device)
            logits = model(x, ptm_ids)
        else:
            logits = model(x)
        prob = torch.softmax(logits, dim=-1)[:, 1]
        all_labels.append(y.cpu().numpy())
        all_probs.append(prob.cpu().numpy())
        all_ptm_ids.append(np.array([m["ptm_id"] for m in meta], dtype=int))
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    ptm_ids = np.concatenate(all_ptm_ids)
    return y_true, y_prob, ptm_ids


def _write_csv(rows: list, csv_path: str):
    fieldnames = ["ptm", "model_checkpoint", "split", "mcc", "f1", "auroc", "aupr", "precision", "recall", "n_rows"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    p = argparse.ArgumentParser(description="Evaluate multi-PTM classifier with per-PTM metrics.")
    p.add_argument("--ptm_types", required=True)
    p.add_argument("--features_root", default="features")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--model_checkpoint", required=True)
    p.add_argument("--model_type", choices=["cnn_gru", "cnn_bilstm"], default="cnn_gru")
    p.add_argument("--condition_on_ptm", action="store_true")
    p.add_argument("--ptm_embed_dim", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--conv_layers", type=int, default=2)
    p.add_argument("--rnn_layers", type=int, default=1)
    p.add_argument("--output_dir", default=None, help="Directory for output files. Default: same dir as checkpoint.")
    args = p.parse_args()

    ptm_types = parse_ptm_list(args.ptm_types)
    device = torch.device(args.device)
    condition = args.condition_on_ptm and args.model_type == "cnn_gru"

    ds = MultiPTMFeaturesDataset(
        features_root=args.features_root,
        ptm_types=ptm_types,
        split=args.split,
    )
    id_to_type = ds.ptm_id_to_type()
    ptm_names = [id_to_type[i] for i in range(len(ptm_types))]

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate,
    )

    model = build_model(args, num_ptm=len(ptm_types)).to(device)
    state = torch.load(args.model_checkpoint, map_location=device)
    model.load_state_dict(state)

    y_true, y_prob, ptm_ids = evaluate_full(model, loader, device, condition)
    overall = compute_binary_metrics(y_true, y_prob)
    per_ptm = metrics_per_ptm(y_true, y_prob, ptm_ids, ptm_names=ptm_names)
    macro = aggregate_macro(per_ptm, ["mcc", "f1", "aupr", "auroc"])

    out_dir = args.output_dir or os.path.dirname(args.model_checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_basename = os.path.basename(args.model_checkpoint)

    # ── Build per-PTM rows (ptmgpt2-style) ──────────────────────────────────
    csv_rows = []
    for name in ptm_names:
        m = per_ptm.get(name, {})
        csv_rows.append({
            "ptm": name,
            "model_checkpoint": ckpt_basename,
            "split": args.split,
            "mcc": m.get("mcc", ""),
            "f1": m.get("f1", ""),
            "auroc": m.get("auroc", ""),
            "aupr": m.get("aupr", ""),
            "precision": m.get("precision", ""),
            "recall": m.get("recall", ""),
            "n_rows": int(m.get("n", 0)),
        })

    # ── Write CSV ────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, f"metrics_{args.split}_per_ptm.csv")
    _write_csv(csv_rows, csv_path)

    # ── Write JSON (full detail including overall + macro) ───────────────────
    json_out = {
        "split": args.split,
        "model_checkpoint": os.path.abspath(args.model_checkpoint),
        "overall": overall,
        "per_ptm": per_ptm,
        "macro": macro,
        "ptm_types": ptm_types,
    }
    json_path = os.path.join(out_dir, f"metrics_{args.split}_per_ptm.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)

    # ── Console output ───────────────────────────────────────────────────────
    header = f"{'ptm':<12} {'mcc':>8} {'f1':>8} {'auroc':>8} {'aupr':>8} {'precision':>10} {'recall':>8} {'n_rows':>8}"
    print(header)
    print("-" * len(header))
    for row in csv_rows:
        def fmt(v):
            if v == "" or v is None:
                return "N/A"
            return f"{float(v):.4f}"
        print(
            f"{row['ptm']:<12} {fmt(row['mcc']):>8} {fmt(row['f1']):>8} "
            f"{fmt(row['auroc']):>8} {fmt(row['aupr']):>8} {fmt(row['precision']):>10} "
            f"{fmt(row['recall']):>8} {row['n_rows']:>8}"
        )
    print()
    print(f"Macro avg  | MCC={macro.get('macro_mcc',0):.4f} | F1={macro.get('macro_f1',0):.4f} "
          f"| AUROC={macro.get('macro_auroc',0):.4f} | AUPR={macro.get('macro_aupr',0):.4f}")
    print()
    print(f"Saved CSV : {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
