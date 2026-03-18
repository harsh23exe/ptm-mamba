import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ptm_classification.datasets import PTMFeaturesDataset
from ptm_classification.metrics import compute_binary_metrics
from ptm_classification.models import CNNGRUClassifier, CNNBiLSTMClassifier


def get_model(model_type: str, num_classes: int = 2) -> torch.nn.Module:
    if model_type == "cnn_gru":
        return CNNGRUClassifier(num_classes=num_classes)
    elif model_type == "cnn_bilstm":
        return CNNBiLSTMClassifier(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        prob = torch.softmax(logits, dim=-1)[:, 1]

        all_labels.append(y.detach().cpu().numpy())
        all_probs.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    metrics = compute_binary_metrics(y_true, y_prob)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PTM classifier on held-out test set."
    )
    parser.add_argument("--ptm_type", default="acet_k")
    parser.add_argument("--features_root", default="features")
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--model_type", choices=["cnn_gru", "cnn_bilstm"], default="cnn_gru")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_root", default="classifier_runs")
    args = parser.parse_args()

    device = torch.device(args.device)

    test_ds = PTMFeaturesDataset(
        features_root=args.features_root,
        ptm_type=args.ptm_type,
        split="test",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = get_model(args.model_type).to(device)
    state = torch.load(args.model_checkpoint, map_location=device)
    model.load_state_dict(state)

    metrics = evaluate(model, test_loader, device)

    # infer run directory from checkpoint path
    run_dir = os.path.dirname(args.model_checkpoint)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "metrics_test.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

