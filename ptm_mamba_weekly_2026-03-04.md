## PTM-Mamba Weekly Summary

### Part 1 – Feature Extraction

- **Goal**: Derive fixed-size PTM-Mamba representations (51×768) for every acet_k window in the train and test sets, without shuffling, so that features remain aligned with labels and can be reused for downstream classifiers and PTM types.

- **Data**:
  - Train: `acet_k/train.csv` — 63,300 rows, columns `Seq`, `Label`, `UniProtID`, `pos`, `full_sequence`.
  - Test: `acet_k/test.csv` — 5,559 rows, same schema.
  - Label distribution:
    - Train: 33,247 negatives (0), 30,053 positives (1) — mildly imbalanced.
    - Test: 2,723 negatives (0), 2,836 positives (1).

- **Procedure**:
  - Implemented `PTMMamba` in `protein_lm/modeling/scripts/infer.py` to wrap the checkpointed PTM-Mamba model and tokenizer.
  - Added `extract_features.py`:
    - Iterates over `acet_k/train.csv` and `acet_k/test.csv` **in file order** using `csv.DictReader` (no shuffling).
    - For each `Seq` (51-aa window), tokenizes with `PTMTokenizer`, builds ESM inputs as needed, and runs PTM-Mamba to obtain `hidden_states` of shape `[seq_len, 768]`.
    - Because some sequences near termini are shorter/longer than 51, the raw `[seq_len, 768]` is padded or truncated along the sequence axis to a fixed `51×768` representation.
    - Each per-row tensor is serialized as float16 bytes, compressed with zlib, base64-encoded, and written into a **single CSV field** `features`.
  - Output layout:
    - `features/acet_k/train.csv.gz`
    - `features/acet_k/test.csv.gz`
    - Each row keeps metadata (`index`, `Label`, `UniProtID`, `pos`, `seq_len`, `hidden_dim`, `dtype`, `encoding`) plus the encoded `features` vector.
  - All paths and PTM type are controlled via CLI (`--ptm_type`, `--data_root`, `--out_root`), so re-running for another PTM only requires changing the PTM name and output root.

- **Result**:
  - We now have compact, PTM-agnostic feature files for acet_k, with one 51×768 PTM-Mamba embedding per window, aligned 1:1 with labels and ready for classifier training.

### Part 2 – Hyperparameter Search for CNN+GRU / CNN+BiLSTM

- **Goal**: Find a strong sequence classifier that consumes the fixed 51×768 embeddings, using CNN+GRU and CNN+BiLSTM architectures, with class weighting and max pooling, and identify good depth (number of CNN and RNN layers).

- **Model and training setup**:
  - Implemented `CNNGRUClassifier` and `CNNBiLSTMClassifier` in `ptm_classification/models/cnn_seq_models.py`:
    - Input: `[batch, 51, 768]`.
    - `LayerNorm` on features → stacked `Conv1d` over sequence (51 as time, 768 as channels).
    - GRU or bidirectional LSTM on the conv outputs.
    - **Adaptive max pooling** over the temporal dimension.
    - Final `LayerNorm` and MLP head → logits `[batch, 2]`.
  - Training logic in `ptm_classification/scripts/train_classifier.py`:
    - Uses `PTMFeaturesDataset` to decode features from `features/acet_k/*.csv.gz`.
    - Stratified train/validation split: 85% train / 15% val.
    - Class weights computed from the train subset and passed to `CrossEntropyLoss` to handle label imbalance.
    - Metrics on validation: MCC, F1, AUROC, precision, recall, accuracy (via `compute_binary_metrics`).
  - Hyperparameter search driver `ptm_classification/scripts/hparam_search.py`:
    - Grid over:
      - `model_type ∈ {cnn_gru, cnn_bilstm}`
      - `conv_layers ∈ {1, 2, 3}`
      - `rnn_layers ∈ {1, 2, 3}`
    - Each configuration trained for up to **35 epochs** (as seen in logs).
    - Launch script: `ptm_classification/run_hparam_search.sbatch` (A100, 24h limit).

- **Key results (validation)**:
  - **Best overall config (by validation MCC)**:
    - **Model**: CNN+GRU
    - **Depth**: `conv_layers = 2`, `rnn_layers = 2`
    - **Best validation MCC**: ~**0.52**
    - **Best validation F1**: ~**0.75**
  - Other observations:
    - Shallower models (1×GRU, 1×conv) converge but plateau at lower MCC (~0.45).
    - Deeper RNN stacks (3 layers) sometimes **collapse to near-random behavior** (MCC ≈ 0) for both CNN+GRU and CNN+BiLSTM, likely due to optimization difficulty and over-parameterization relative to dataset size.
    - CNN+BiLSTM variants reach MCC in the high 0.4s (best ~0.49) but do not surpass the best CNN+GRU 2×2 setup on validation.

- **Why choose CNN+GRU with 2 conv and 2 GRU layers**:
  - It achieves the **highest validation MCC** and strong F1 among all tested configurations, indicating the best balance between precision and recall on the imbalanced acet_k task.
  - It is **simpler and more stable** than the deeper 3-layer RNN configurations, which showed signs of saturation or collapse (flat loss and MCC ≈ 0 for many epochs).
  - Compared to CNN+BiLSTM, CNN+GRU 2×2 offers:
    - Slightly better validation MCC and F1.
    - Lower computational cost (unidirectional GRU vs. bidirectional LSTM) for the same number of layers, which is important when scaling to more PTM types.

- **Next steps (optional)**:
  - Evaluate the selected CNN+GRU (2 conv, 2 GRU layers) on the held-out **test set** using `eval_classifier.py` to report test MCC, F1, AUROC, precision, and recall.
  - Consider a narrower second-stage search around the best region (e.g., tuning hidden size, dropout, learning rate) once the architecture choice is fixed.

