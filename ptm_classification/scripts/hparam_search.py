import argparse
from copy import deepcopy
from itertools import product

from ptm_classification.scripts.train_classifier import run_training


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search over CNN/LSTM layer counts for PTM classifiers."
    )
    parser.add_argument("--ptm_type", default="acet_k")
    parser.add_argument("--features_root", default="features")
    parser.add_argument("--output_root", default="classifier_runs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")

    # Search spaces
    parser.add_argument(
        "--model_types",
        nargs="+",
        default=["cnn_gru", "cnn_bilstm"],
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

    args = parser.parse_args()

    base = argparse.Namespace(
        ptm_type=args.ptm_type,
        features_root=args.features_root,
        output_root=args.output_root,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_size=args.val_size,
        seed=args.seed,
        device=args.device,
        # placeholders to be overwritten in the loop
        model_type=None,
        conv_layers=None,
        rnn_layers=None,
    )

    for model_type, conv_layers, rnn_layers in product(
        args.model_types, args.conv_layers_list, args.rnn_layers_list
    ):
        exp_args = deepcopy(base)
        exp_args.model_type = model_type
        exp_args.conv_layers = conv_layers
        exp_args.rnn_layers = rnn_layers

        print(
            f"\n=== Running config: model={model_type}, "
            f"conv_layers={conv_layers}, rnn_layers={rnn_layers}, "
            f"epochs={exp_args.num_epochs} ==="
        )
        run_dir = run_training(exp_args)
        print(f"Finished config: model={model_type}, conv_layers={conv_layers}, "
              f"rnn_layers={rnn_layers}. Run dir: {run_dir}")


if __name__ == "__main__":
    main()

