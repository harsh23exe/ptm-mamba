#!/usr/bin/env python3
"""
Build one combined JSON per PTM under classifier_runs/.

Each file lists cnn_gru and cnn_bilstm runs with:
  - config (hyperparameters)
  - final_epoch / final_metrics_val / final_val_mcc (last epoch in metrics_val.json only)

Output: classifier_runs/<ptm_type>/<ptm_type>_cnn_gru_bilstm_combined.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_TYPES = ("cnn_gru", "cnn_bilstm")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_runs_for_model(model_root: Path, repo_root: Path) -> list[dict]:
    runs: list[dict] = []
    if not model_root.is_dir():
        return runs

    for run_dir in sorted((p for p in model_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        cfg_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics_val.json"
        if not cfg_path.exists():
            continue

        cfg = load_json(cfg_path)
        rel_run = run_dir.relative_to(repo_root)
        run_entry: dict = {
            "timestamp": run_dir.name,
            "run_dir": str(rel_run).replace("\\", "/"),
            "config": cfg,
        }

        if metrics_path.exists():
            metrics = load_json(metrics_path)
            if isinstance(metrics, list) and len(metrics) > 0:
                last = metrics[-1]
                final_val = (last or {}).get("val")
                run_entry["final_epoch"] = last.get("epoch")
                run_entry["final_metrics_val"] = final_val
                run_entry["final_val_mcc"] = (final_val or {}).get("mcc")
            else:
                run_entry["final_epoch"] = None
                run_entry["final_metrics_val"] = None
                run_entry["final_val_mcc"] = None
        else:
            run_entry["final_epoch"] = None
            run_entry["final_metrics_val"] = None
            run_entry["final_val_mcc"] = None

        runs.append(run_entry)
    return runs


def best_run_by_final_mcc(runs: list[dict]) -> dict | None:
    best: dict | None = None
    for r in runs:
        mcc = r.get("final_val_mcc")
        if mcc is None:
            continue
        if best is None or mcc > best["final_val_mcc"]:
            best = r
    return best


def build_ptm_summary(ptm_dir: Path, repo_root: Path) -> dict:
    ptm_type = ptm_dir.name
    rel_ptm = ptm_dir.relative_to(repo_root)
    base = str(rel_ptm).replace("\\", "/")

    combined: dict = {
        "ptm_type": ptm_type,
        "source_root": base,
        "generated_from": f"{base}/<model_type>/<timestamp>/ (final epoch metrics only)",
        "model_types": {},
    }

    for model_type in MODEL_TYPES:
        model_root = ptm_dir / model_type
        runs = collect_runs_for_model(model_root, repo_root)
        combined["model_types"][model_type] = {
            "runs": runs,
            "best_run_by_final_val_mcc": best_run_by_final_mcc(runs),
            "num_runs": len(runs),
            "num_runs_with_final_metrics": sum(
                1 for r in runs if r.get("final_metrics_val") is not None
            ),
        }

    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classifier-runs-root",
        type=Path,
        default=Path("classifier_runs"),
        help="Path to classifier_runs (default: ./classifier_runs)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root for relative run_dir paths (default: cwd)",
    )
    args = parser.parse_args()

    repo_root = (args.repo_root or Path.cwd()).resolve()
    cr = (args.classifier_runs_root if args.classifier_runs_root.is_absolute() else repo_root / args.classifier_runs_root).resolve()

    if not cr.is_dir():
        raise SystemExit(f"Not a directory: {cr}")

    for ptm_dir in sorted(p for p in cr.iterdir() if p.is_dir()):
        has_models = any((ptm_dir / mt).is_dir() for mt in MODEL_TYPES)
        if not has_models:
            continue

        out_name = f"{ptm_dir.name}_cnn_gru_bilstm_combined.json"
        out_path = ptm_dir / out_name
        summary = build_ptm_summary(ptm_dir, repo_root)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        print(f"Wrote {out_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
