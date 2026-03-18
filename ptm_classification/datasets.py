import base64
import csv
import gzip
import os
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PTMFeaturesDataset(Dataset):
    """
    Dataset over PTM-Mamba feature CSVs.

    Expects files at:
        features/<ptm_type>/<split>.csv.gz

    Each row must contain:
        index, Label, UniProtID, pos, seq_len, hidden_dim, dtype, encoding, features
    where `features` is base64(zlib(float16_bytes)) of a [seq_len, hidden_dim] tensor.
    """

    def __init__(
        self,
        features_root: str,
        ptm_type: str,
        split: str,
        expected_seq_len: int = 51,
        expected_hidden_dim: int = 768,
    ) -> None:
        super().__init__()

        path = os.path.join(features_root, ptm_type, f"{split}.csv.gz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Features file not found: {path}")

        self._path = path
        self._expected_seq_len = expected_seq_len
        self._expected_hidden_dim = expected_hidden_dim

        self._labels: List[int] = []
        self._features_str: List[str] = []
        self._meta: List[Tuple[str, str]] = []  # (UniProtID, pos)
        self._seq_lens: List[int] = []
        self._hidden_dims: List[int] = []

        with gzip.open(path, "rt", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                self._labels.append(int(row["Label"]))
                self._features_str.append(row["features"])
                self._meta.append((row.get("UniProtID", ""), row.get("pos", "")))
                # fall back to expected values if columns are missing
                self._seq_lens.append(int(row.get("seq_len", self._expected_seq_len)))
                self._hidden_dims.append(int(row.get("hidden_dim", self._expected_hidden_dim)))

        if len(self._labels) == 0:
            raise RuntimeError(f"No rows found in features file: {path}")

    def __len__(self) -> int:
        return len(self._labels)

    def _decode_features(self, idx: int) -> torch.Tensor:
        s = self._features_str[idx]
        raw = base64.b64decode(s)
        import zlib  # local import to avoid cost if unused

        decomp = zlib.decompress(raw)
        arr = np.frombuffer(decomp, dtype=np.float16)

        seq_len = self._seq_lens[idx]
        hidden_dim = self._hidden_dims[idx]

        # If metadata and actual size disagree, try to infer seq_len.
        if seq_len * hidden_dim != arr.size:
            if hidden_dim == self._expected_hidden_dim and arr.size % hidden_dim == 0:
                seq_len = arr.size // hidden_dim
            else:
                raise ValueError(
                    f"Unexpected feature size {arr.size} for index {idx}: "
                    f"seq_len={seq_len}, hidden_dim={hidden_dim}"
                )

        # reshape to the true [seq_len, hidden_dim]
        arr = arr.reshape(seq_len, hidden_dim).astype(np.float32)

        # pad or truncate along sequence dimension to fixed expected_seq_len
        if seq_len < self._expected_seq_len:
            pad_len = self._expected_seq_len - seq_len
            pad = np.zeros((pad_len, hidden_dim), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif seq_len > self._expected_seq_len:
            arr = arr[: self._expected_seq_len, :]

        tensor = torch.from_numpy(arr)  # [expected_seq_len, hidden_dim] float32
        return tensor

    def __getitem__(self, idx: int):
        x = self._decode_features(idx)
        y = torch.tensor(self._labels[idx], dtype=torch.long)
        uniprot_id, pos = self._meta[idx]
        return x, y, {"UniProtID": uniprot_id, "pos": pos, "index": idx}


