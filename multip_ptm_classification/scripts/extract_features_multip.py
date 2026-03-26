"""
Extract PTM-Mamba hidden states for one PTM folder, optionally injecting the
dataset-specific PTM token at the window center before forwarding.

Output layout matches protein_lm/modeling/scripts/extract_features.py:
    <out_root>/<ptm_type>/train.csv.gz and test.csv.gz

This lets each PTM use the correct `<...>` spelling in Mamba input while
keeping separate output directories; multip training then pools those shards.
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from protein_lm.modeling.scripts.infer import PTMMamba  # noqa: E402
from multip_ptm_classification.sequence_inject import (  # noqa: E402
    inject_ptm_token_at_center,
)
from multip_ptm_classification.ptm_token_map import get_ptm_token_for_folder  # noqa: E402


def _encode_tensor(vec_2d: torch.Tensor) -> str:
    import numpy as np

    arr = vec_2d.contiguous().to(dtype=torch.float16, device="cpu").numpy()
    raw = arr.tobytes(order="C")
    comp = zlib.compress(raw, level=3)
    return base64.b64encode(comp).decode("ascii")


def _maybe_transform_seq(row, args, ptm_token: str) -> str:
    seq = row["Seq"]
    if args.inject == "none":
        return seq
    if args.inject == "center":
        return inject_ptm_token_at_center(seq, ptm_token)
    raise ValueError(f"Unknown --inject mode: {args.inject}")


def _write_split_csv(model: PTMMamba, csv_path: str, out_csv_gz: str, batch_size: int, args, ptm_token: str):
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

            hs = outputs.hidden_states
            if hs is None:
                raise RuntimeError("Model returned None for hidden_states.")

            for j in range(len(batch_rows)):
                tok_len = len(input_ids_list[j])
                vec = hs[j, :tok_len, :].cpu()
                features_str = _encode_tensor(vec)

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
            batch_seqs.append(_maybe_transform_seq(row, args, ptm_token))
            if len(batch_rows) >= batch_size:
                flush_batch()

        flush_batch()

    print(f"    Wrote: {out_csv_gz}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract PTMMamba features with optional PTM-token injection."
    )
    parser.add_argument("--ptm_type", required=True)
    parser.add_argument("--ckpt_path", default="ckpt/best.ckpt")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--out_root", default="features")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument(
        "--inject",
        default="center",
        choices=["none", "center"],
        help="Replace center AA with PTM token (center), or use raw Seq (none).",
    )
    parser.add_argument(
        "--ptm_token",
        default=None,
        help="Override tokenizer string for this PTM (must match PTMTokenizer vocab).",
    )
    args = parser.parse_args()

    data_root = args.data_root or os.path.join(_PROJECT_ROOT, args.ptm_type)
    out_root = os.path.join(_PROJECT_ROOT, args.out_root, args.ptm_type)
    ptm_token = get_ptm_token_for_folder(args.ptm_type, override=args.ptm_token)

    print(f"PTM folder : {args.ptm_type}")
    print(f"PTM token  : {ptm_token}")
    print(f"Inject mode: {args.inject}")
    print(f"Data root : {data_root}")
    print(f"Output    : {out_root}")
    print()

    print("Loading PTMMamba model ...")
    model = PTMMamba(args.ckpt_path, device=args.device)
    print("Model loaded.\n")

    for split in args.splits:
        csv_path = os.path.join(data_root, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"[WARNING] {csv_path} not found, skipping split '{split}'.")
            continue
        print(f"Processing split: {split}  ({csv_path})")
        out_csv_gz = os.path.join(out_root, f"{split}.csv.gz")
        _write_split_csv(model, csv_path, out_csv_gz, args.batch_size, args, ptm_token)
        print("  Done.\n")

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
