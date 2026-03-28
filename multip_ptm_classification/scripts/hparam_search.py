import argparse
import json
import os
import sys
from copy import deepcopy
from itertools import product

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from multip_ptm_classification.scripts.train_multip import run_training


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for multi-PTM classifier."
    )

    parser.add_argument(
        "--ptm_types",
        required=True,
        help="Comma-separated PTM folder names, e.g. acet_k,met_r,phos_y,sumo_k",
    )
    parser.add_argument("--features_root", default="features")
    parser.add_argument("--output_root", default="classifier_runs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--condition_on_ptm", action="store_true")
    parser.add_argument("--ptm_embed_dim", type=int, default=32)
    parser.add_argument("--balance_ptm_batches", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)

    # Search spaces
    parser.add_argument(
        "--model_types",
        nargs="+",
        default=["cnn_gru"],
        help="Model types to search over.",
    )
    parser.add_argument(
        "--conv_layers_list",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Candidate numbers of CNN layers.",
    )
    parser.add_argument(
        "--rnn_layers_list",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Candidate numbers of GRU/LSTM layers.",
    )
    parser.add_argument(
        "--lr_list",
        nargs="+",
        type=float,
        default=None,
        help="Learning rates to search. If omitted, uses --lr only.",
    )
    parser.add_argument(
        "--ptm_embed_dim_list",
        nargs="+",
        type=int,
        default=None,
        help="PTM embed dims to search (only with --condition_on_ptm). If omitted, uses --ptm_embed_dim only.",
    )

    args = parser.parse_args()

    lr_list = args.lr_list if args.lr_list else [args.lr]
    ptm_embed_dim_list = args.ptm_embed_dim_list if args.ptm_embed_dim_list else [args.ptm_embed_dim]

    base = argparse.Namespace(
        ptm_types=args.ptm_types,
        features_root=args.features_root,
        output_root=args.output_root,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        val_size=args.val_size,
        seed=args.seed,
        device=args.device,
        condition_on_ptm=args.condition_on_ptm,
        balance_ptm_batches=args.balance_ptm_batches,
        amp=args.amp,
        data_parallel=args.data_parallel,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        model_type=None,
        conv_layers=None,
        rnn_layers=None,
        lr=None,
        ptm_embed_dim=None,
    )

    combos = list(product(
        args.model_types,
        args.conv_layers_list,
        args.rnn_layers_list,
        lr_list,
        ptm_embed_dim_list,
    ))

    total = len(combos)
    summary = []

    print(f"Total configurations to search: {total}")
    print(f"PTM types: {args.ptm_types}")
    print(f"Model types: {args.model_types}")
    print(f"Conv layers: {args.conv_layers_list}")
    print(f"RNN layers: {args.rnn_layers_list}")
    print(f"LR: {lr_list}")
    print(f"PTM embed dim: {ptm_embed_dim_list}")
    print()

    for i, (model_type, conv_layers, rnn_layers, lr, ptm_embed_dim) in enumerate(combos, 1):
        if model_type != "cnn_gru" and args.condition_on_ptm:
            print(f"[{i}/{total}] SKIP: condition_on_ptm not supported for {model_type}")
            continue

        exp_args = deepcopy(base)
        exp_args.model_type = model_type
        exp_args.conv_layers = conv_layers
        exp_args.rnn_layers = rnn_layers
        exp_args.lr = lr
        exp_args.ptm_embed_dim = ptm_embed_dim

        print(
            f"\n[{i}/{total}] model={model_type}, conv={conv_layers}, rnn={rnn_layers}, "
            f"lr={lr}, ptm_embed_dim={ptm_embed_dim}"
        )
        print("=" * 70)

        try:
            run_dir = run_training(exp_args)
            print(f"Finished. Run dir: {run_dir}")

            metrics_json = os.path.join(run_dir, "metrics_test_per_ptm.json")
            if os.path.isfile(metrics_json):
                with open(metrics_json) as f:
                    results = json.load(f)
                macro = results.get("macro", {})
            else:
                macro = {}

            summary.append({
                "model_type": model_type,
                "conv_layers": conv_layers,
                "rnn_layers": rnn_layers,
                "lr": lr,
                "ptm_embed_dim": ptm_embed_dim,
                "run_dir": run_dir,
                "macro_mcc": macro.get("macro_mcc"),
                "macro_f1": macro.get("macro_f1"),
                "macro_auroc": macro.get("macro_auroc"),
                "macro_aupr": macro.get("macro_aupr"),
            })
        except Exception as e:
            print(f"FAILED: {e}")
            summary.append({
                "model_type": model_type,
                "conv_layers": conv_layers,
                "rnn_layers": rnn_layers,
                "lr": lr,
                "ptm_embed_dim": ptm_embed_dim,
                "run_dir": None,
                "error": str(e),
            })

    # Write summary
    summary_dir = os.path.join(args.output_root, "all_ptms_model", "hparam_search")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "hparam_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 70)
    print(f"\n{'model':<14} {'conv':>5} {'rnn':>5} {'lr':>10} {'embed':>6} {'macro_MCC':>10} {'macro_F1':>9} {'macro_AUROC':>12} {'macro_AUPR':>11}")
    print("-" * 90)
    for s in summary:
        if s.get("run_dir") is None:
            print(f"{s['model_type']:<14} {s['conv_layers']:>5} {s['rnn_layers']:>5} {s['lr']:>10.1e} {s['ptm_embed_dim']:>6}   FAILED")
            continue
        def fmt(v):
            return f"{v:.4f}" if v is not None else "N/A"
        print(
            f"{s['model_type']:<14} {s['conv_layers']:>5} {s['rnn_layers']:>5} {s['lr']:>10.1e} {s['ptm_embed_dim']:>6} "
            f"{fmt(s.get('macro_mcc')):>10} {fmt(s.get('macro_f1')):>9} "
            f"{fmt(s.get('macro_auroc')):>12} {fmt(s.get('macro_aupr')):>11}"
        )

    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
