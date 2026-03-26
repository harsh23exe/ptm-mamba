import base64
import csv
import gzip
import os
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiPTMFeaturesDataset(Dataset):
    """
    Concatenate several per-PTM feature shards into one dataset.

    Expects the same CSV layout as ptm_classification.datasets.PTMFeaturesDataset:
        index, Label, UniProtID, pos, seq_len, hidden_dim, dtype, encoding, features

    Each sample returns:
        features: FloatTensor [expected_seq_len, expected_hidden_dim]
        label: LongTensor scalar
        meta: dict with UniProtID, pos, index, ptm_type (str), ptm_id (int)
    """

    def __init__(
        self,
        features_root: str,
        ptm_types: List[str],
        split: str,
        expected_seq_len: int = 51,
        expected_hidden_dim: int = 768,
        ptm_type_to_id: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        if not ptm_types:
            raise ValueError("ptm_types must be non-empty")

        self._expected_seq_len = expected_seq_len
        self._expected_hidden_dim = expected_hidden_dim
        self._ptm_types = list(ptm_types)
        self._ptm_type_to_id: Dict[str, int]
        if ptm_type_to_id is not None:
            self._ptm_type_to_id = dict(ptm_type_to_id)
        else:
            self._ptm_type_to_id = {p: i for i, p in enumerate(self._ptm_types)}

        self._labels: List[int] = []
        self._features_str: List[str] = []
        self._meta: List[Tuple[str, str]] = []
        self._seq_lens: List[int] = []
        self._hidden_dims: List[int] = []
        self._ptm_ids: List[int] = []
        self._row_index: List[int] = []

        global_i = 0
        for ptm in self._pid_order():
            path = os.path.join(features_root, ptm, f"{split}.csv.gz")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Features for PTM '{ptm}' split '{split}' not found: {path}"
                )
            pid = self._ptm_type_to_id[ptm]
            with gzip.open(path, "rt", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    self._labels.append(int(row["Label"]))
                    self._features_str.append(row["features"])
                    self._meta.append((row.get("UniProtID", ""), row.get("pos", "")))
                    self._seq_lens.append(
                        int(row.get("seq_len", self._expected_seq_len))
                    )
                    self._hidden_dims.append(
                        int(row.get("hidden_dim", self._expected_hidden_dim))
                    )
                    self._ptm_ids.append(pid)
                    if row.get("index", "") != "":
                        self._row_index.append(int(row["index"]))
                    else:
                        self._row_index.append(global_i)
                    global_i += 1

        if len(self._labels) == 0:
            raise RuntimeError(
                f"No rows loaded for split '{split}' from PTM types {ptm_types}"
            )

    def _pid_order(self) -> List[str]:
        return self._ptm_types

    def __len__(self) -> int:
        return len(self._labels)

    def ptm_types(self) -> List[str]:
        return list(self._ptm_types)

    def ptm_id_to_type(self) -> Dict[int, str]:
        inv = {v: k for k, v in self._ptm_type_to_id.items()}
        return {i: inv[i] for i in range(len(self._ptm_types))}

    def _decode_features(self, idx: int) -> torch.Tensor:
        s = self._features_str[idx]
        raw = base64.b64decode(s)
        decomp = zlib.decompress(raw)
        arr = np.frombuffer(decomp, dtype=np.float16)

        seq_len = self._seq_lens[idx]
        hidden_dim = self._hidden_dims[idx]

        if seq_len * hidden_dim != arr.size:
            if hidden_dim == self._expected_hidden_dim and arr.size % hidden_dim == 0:
                seq_len = arr.size // hidden_dim
            else:
                raise ValueError(
                    f"Unexpected feature size {arr.size} for index {idx}: "
                    f"seq_len={seq_len}, hidden_dim={hidden_dim}"
                )

        arr = arr.reshape(seq_len, hidden_dim).astype(np.float32)

        if seq_len < self._expected_seq_len:
            pad_len = self._expected_seq_len - seq_len
            pad = np.zeros((pad_len, hidden_dim), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif seq_len > self._expected_seq_len:
            arr = arr[: self._expected_seq_len, :]

        return torch.from_numpy(arr)

    def __getitem__(self, idx: int):
        x = self._decode_features(idx)
        y = torch.tensor(self._labels[idx], dtype=torch.long)
        uniprot_id, pos = self._meta[idx]
        ptm_type = self._ptm_types[self._ptm_ids[idx]]
        meta = {
            "UniProtID": uniprot_id,
            "pos": pos,
            "index": self._row_index[idx],
            "ptm_type": ptm_type,
            "ptm_id": self._ptm_ids[idx],
        }
        return x, y, meta
