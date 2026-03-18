"""
Feature extraction pipeline for PTM classification.

Iterates over train and test splits of a PTM dataset, passes each window
sequence through PTMMamba, and saves the resulting hidden-state tensor
(shape: seq_len x hidden_dim, e.g. 51 x 768) alongside its label.

Output layout
--------------
features/<ptm_type>/
    train.csv.gz   # one row per window, features stored as a single field
    test.csv.gz

Usage
-----
python protein_lm/modeling/scripts/extract_features.py \
    --ptm_type acet_k \
    --ckpt_path ckpt/best.ckpt \
    --data_root acet_k \
    --out_root  features \
    --batch_size 32 \
    --device cuda:0

CSV encoding
------------
The 51x768 tensor is stored in a single CSV field `features` using:
- float16 (little-endian) bytes
- zlib compression
- base64 encoding

Decoding example:

    import base64, zlib, numpy as np
    raw = zlib.decompress(base64.b64decode(row["features"]))
    arr = np.frombuffer(raw, dtype=np.float16).reshape(51, 768)
"""

import argparse
import base64
import csv
import gzip
import os
import sys
import zlib

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# ── resolve project root so the script is runnable from any cwd ──────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from protein_lm.modeling.scripts.infer import PTMMamba  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_split(csv_path: str):
    """Return (seqs, labels, meta_rows) without shuffling."""
    seqs, labels, meta_rows = [], [], []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            seqs.append(row["Seq"])
            labels.append(int(row["Label"]))
            meta_rows.append({
                "UniProtID": row.get("UniProtID", ""),
                "pos":       row.get("pos", ""),
                "Label":     row["Label"],
            })
    return seqs, labels, meta_rows


def _encode_tensor_51x768(vec_2d: torch.Tensor) -> str:
    """
    Encode a [seq_len, hidden_dim] tensor into a single CSV-safe string.

    Returns base64(zlib(float16_bytes)).
    """
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "NumPy is required to write features to CSV efficiently. "
            "Install NumPy in the runtime environment."
        ) from e

    arr = vec_2d.contiguous().to(dtype=torch.float16, device="cpu").numpy()  # type: ignore[attr-defined]
    raw = arr.tobytes(order="C")
    comp = zlib.compress(raw, level=3)
    return base64.b64encode(comp).decode("ascii")


def _write_split_csv(model: PTMMamba, csv_path: str, out_csv_gz: str, batch_size: int):
    """
    Stream through the input CSV in row order, run inference in mini-batches,
    and write one output CSV row per input row (no shuffling).
    """
    os.makedirs(os.path.dirname(out_csv_gz), exist_ok=True)

    fieldnames = [
        "index",
        "Label",
        "UniProtID",
        "pos",
        "seq_len",
        "hidden_dim",
        "dtype",
        "encoding",
        "features",
    ]

    with open(csv_path, newline="") as fh_in, gzip.open(out_csv_gz, "wt", newline="") as fh_out:
        reader = csv.DictReader(fh_in)
        writer = csv.DictWriter(fh_out, fieldnames=fieldnames)
        writer.writeheader()

        idx = 0
        batch_rows = []
        batch_seqs = []

        def flush_batch():
            nonlocal idx, batch_rows, batch_seqs
            if not batch_rows:
                return

            input_ids_list = model.tokenizer(batch_seqs)
            padded = pad_sequence(
                [torch.tensor(x) for x in input_ids_list],
                batch_first=True,
                padding_value=model.tokenizer.pad_token_id,
            ).to(model.device)

            with torch.no_grad():
                outputs = model._infer(padded)

            hs = outputs.hidden_states  # type: ignore[union-attr]
            if hs is None:
                raise RuntimeError(
                    "Model returned None for hidden_states. "
                    "Ensure the checkpoint was trained with output_hidden_states=True."
                )

            for j in range(len(batch_rows)):
                tok_len = len(input_ids_list[j])
                vec = hs[j, :tok_len, :].cpu()  # [tok_len, hidden_dim]
                features_str = _encode_tensor_51x768(vec)

                r = batch_rows[j]
                writer.writerow({
                    "index": idx,
                    "Label": int(r["Label"]),
                    "UniProtID": r.get("UniProtID", ""),
                    "pos": r.get("pos", ""),
                    "seq_len": int(vec.shape[0]),
                    "hidden_dim": int(vec.shape[1]),
                    "dtype": "float16",
                    "encoding": "b64+zlib",
                    "features": features_str,
                })
                idx += 1

            batch_rows = []
            batch_seqs = []

        for row in tqdm(reader, desc="  rows", leave=False):
            batch_rows.append(row)
            batch_seqs.append(row["Seq"])
            if len(batch_rows) >= batch_size:
                flush_batch()

        flush_batch()

    print(f"    Wrote: {out_csv_gz}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract PTMMamba hidden-state features for a PTM dataset."
    )
    parser.add_argument("--ptm_type",   required=True,
                        help="PTM type folder name, e.g. acet_k  (change this to switch PTMs)")
    parser.add_argument("--ckpt_path",  default="ckpt/best.ckpt",
                        help="Path to the PTMMamba checkpoint.")
    parser.add_argument("--data_root",  default=None,
                        help="Root directory that contains <ptm_type>/train.csv and test.csv. "
                             "Defaults to <project_root>/<ptm_type>.")
    parser.add_argument("--out_root",   default="features",
                        help="Root directory for output feature files.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Sequences per forward pass.")
    parser.add_argument("--device",     default="cuda:0",
                        help="Torch device string.")
    parser.add_argument("--splits",     nargs="+", default=["train", "test"],
                        choices=["train", "test"],
                        help="Which splits to process.")
    args = parser.parse_args()

    data_root = args.data_root or os.path.join(_PROJECT_ROOT, args.ptm_type)
    out_root  = os.path.join(_PROJECT_ROOT, args.out_root, args.ptm_type)

    print(f"PTM type  : {args.ptm_type}")
    print(f"Data root : {data_root}")
    print(f"Output    : {out_root}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Device    : {args.device}")
    print(f"Batch size: {args.batch_size}")
    print()

    # ── load model once ───────────────────────────────────────────────────────
    print("Loading PTMMamba model ...")
    model = PTMMamba(args.ckpt_path, device=args.device)
    print("Model loaded.\n")

    # ── process each split ────────────────────────────────────────────────────
    for split in args.splits:
        csv_path = os.path.join(data_root, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"[WARNING] {csv_path} not found, skipping split '{split}'.")
            continue

        print(f"Processing split: {split}  ({csv_path})")
        out_csv_gz = os.path.join(out_root, f"{split}.csv.gz")
        _write_split_csv(model, csv_path, out_csv_gz, args.batch_size)
        print("  Done.\n")

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
