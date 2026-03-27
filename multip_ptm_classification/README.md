# Multi-PTM Classification

A single binary classifier trained on pooled data from multiple PTM types, using PTM-Mamba hidden states as input features. The model is optionally conditioned on PTM type so it can learn shared representations across PTMs while still accounting for PTM-specific patterns.

## Architecture

```
Input: PTM-Mamba hidden states [batch, 51, 768]
                    |
              LayerNorm
                    |
         Conv1D stack (N layers, kernel=3, channels=256)
                    |
                  GRU
                    |
         AdaptiveMaxPool1d -> [batch, hidden]
                    |
          concat with PTM embedding (32-d)    <-- optional, via --condition_on_ptm
                    |
           Linear -> ReLU -> Dropout -> Linear -> [batch, 2]
```

The backbone is `CNNGRUClassifier` from `ptm_classification`. The conditioned variant (`PTMConditionedCNNGRU`) adds a learned embedding per PTM type and concatenates it with the pooled GRU output before the classification head. This gives the model a signal about which PTM it is classifying, without requiring separate parameters per PTM.

The classifier does **not** run Mamba inference. It consumes pre-extracted features stored as compressed CSVs.

## Data Format

### Raw data (per PTM)

Each PTM has a directory under `datasets/` with `train.csv` and `test.csv`:

```
datasets/acet_k/train.csv
datasets/acet_k/test.csv
datasets/met_r/train.csv
...
```

Columns: `Seq`, `Label` (0/1), `UniProtID`, `pos`, `full_sequence`.
`Seq` is a 51-residue window centered on the candidate modification site.

### Features (per PTM)

Feature extraction passes each `Seq` through PTM-Mamba and stores the hidden state tensor (51 x 768) in a compressed CSV:

```
features/acet_k/train.csv.gz
features/acet_k/test.csv.gz
features/met_r/train.csv.gz
...
```

Columns: `index`, `Label`, `UniProtID`, `pos`, `seq_len`, `hidden_dim`, `dtype`, `encoding`, `features`.
The `features` field is `base64(zlib(float16_bytes))` of shape `[seq_len, 768]`.

### PTM token injection

Before feature extraction, the center amino acid of each 51-residue window can be replaced with the PTM-specific token recognized by PTM-Mamba's tokenizer:

| PTM folder | Token |
|------------|-------|
| `acet_k` | `<N6-acetyllysine>` |
| `phos_s` | `<Phosphoserine>` |
| `phos_t` | `<Phosphothreonine>` |
| `phos_y` | `<Phosphotyrosine>` |
| `met_r` | `<Omega-N-methylarginine>` |
| `sumo_k` | `<N6-succinyllysine>` |

Full mapping is in `ptm_token_map.py`. Override per run with `--ptm_token`.

## Directory Layout

```
multip_ptm_classification/
    __init__.py
    datasets.py              # MultiPTMFeaturesDataset (pools shards from N PTMs)
    metrics.py               # Per-PTM + macro MCC, F1, AUPR, AUROC
    ptm_token_map.py         # folder name -> tokenizer string
    sequence_inject.py       # inject PTM token at window center
    models/
        __init__.py
        conditioned_cnn_gru.py   # PTMConditionedCNNGRU
    scripts/
        extract_features_multip.py  # feature extraction with token injection
        train_multip.py             # training + automatic test eval
        eval_multip.py              # standalone evaluation
        compare_to_single_ptm.py    # tabulate multi vs single-PTM results
    run_train_multip_2xa100.sbatch  # Slurm launcher for 2xA100
```

Checkpoints and metrics go to:

```
classifier_runs/all_ptms_model/<model_type>/<timestamp>/
    config.json
    model_best.pt
    metrics_val.json
    metrics_test_per_ptm.csv
    metrics_test_per_ptm.json
```

## How to Train

### 1. Extract features (once per PTM)

Run inside Apptainer on a GPU node:

```bash
python -m multip_ptm_classification.scripts.extract_features_multip \
    --ptm_type acet_k \
    --ckpt_path ckpt/best.ckpt \
    --data_root datasets/acet_k \
    --out_root features \
    --inject center \
    --device cuda:0
```

Repeat for each PTM folder.

### 2. Train the multi-PTM model

Interactive test (short run):

```bash
python -m multip_ptm_classification.scripts.train_multip \
    --ptm_types "acet_k,met_r,phos_y,sumo_k" \
    --features_root features \
    --output_root classifier_runs \
    --model_type cnn_gru \
    --condition_on_ptm \
    --balance_ptm_batches \
    --batch_size 64 \
    --num_epochs 3 \
    --lr 1e-3 \
    --device cuda:0 \
    --amp
```

Full training via Slurm (2xA100):

```bash
sbatch multip_ptm_classification/run_train_multip_2xa100.sbatch
```

Training automatically evaluates the best checkpoint on the test set when finished and writes per-PTM metrics.

### Key flags

| Flag | Effect |
|------|--------|
| `--condition_on_ptm` | Concatenate a learned PTM embedding before the head |
| `--balance_ptm_batches` | Weighted sampler so each PTM is drawn equally often |
| `--amp` | FP16 mixed precision (recommended on A100) |
| `--data_parallel` | Use both GPUs via `DataParallel` |
| `--grad_accum_steps N` | Accumulate gradients over N steps for larger effective batch |

## Metrics

Evaluated per PTM on the held-out test set:

- **MCC** (Matthews Correlation Coefficient) -- primary selection metric during training
- **F1**
- **AUROC**
- **AUPR** (Average Precision)
- **Precision**, **Recall**

Output is saved as both CSV and JSON. Console prints a table like:

```
ptm              mcc       f1    auroc     aupr  precision   recall   n_rows
------------------------------------------------------------------------
acet_k         0.3095   0.6953   0.7191   0.7129     0.6282   0.7786     5559
phos_y         0.3318   0.4405   0.7909   0.4535     0.4720   0.4129     5814
sumo_k         0.3388   0.6619   0.7350   0.6535     0.5594   0.8106     8615
met_r          0.3351   0.4813   0.7506   0.5008     0.4811   0.4816     4449
```

## Comparing to Single-PTM Models

```bash
python -m multip_ptm_classification.scripts.compare_to_single_ptm \
    --multip_json classifier_runs/all_ptms_model/cnn_gru/<timestamp>/metrics_test_per_ptm.json \
    --classifier_root classifier_runs
```

Prints a side-by-side table of multi-PTM vs latest per-PTM model metrics.

## Single-PTM vs Multi-PTM Classifier

### Input differences

The two classifiers consume the same type of tensor -- PTM-Mamba hidden states of shape `[batch, 51, 768]` -- but the data behind them differs in scope and composition.

**Single-PTM classifier** (`ptm_classification/`):
- Trained and evaluated on one PTM at a time (e.g. only `acet_k`).
- Each training run loads `features/acet_k/train.csv.gz` alone.
- The model only ever sees positive/negative examples for that one modification type.

**Multi-PTM classifier** (`multip_ptm_classification/`):
- Pools feature shards from all PTMs into one dataset (e.g. `acet_k` + `met_r` + `phos_y` + `sumo_k` combined).
- Each sample carries a `ptm_id` integer that tracks which PTM it belongs to.
- The model trains on a mix of all PTMs simultaneously.

### Architecture differences

The CNN and GRU backbone is identical between both. The difference is in the classification head and how PTM identity is handled.

```
                      Single-PTM (CNNGRUClassifier)
                      ─────────────────────────────
                      pooled [B, 256]
                            |
                      Linear(256, 256) -> ReLU -> Dropout
                            |
                      Linear(256, 2)  ->  logits


                      Multi-PTM conditioned (PTMConditionedCNNGRU)
                      ────────────────────────────────────────────
                      pooled [B, 256]     PTM embedding [B, 32]
                            \                  /
                             concat [B, 288]
                                  |
                            Linear(288, 256) -> ReLU -> Dropout
                                  |
                            Linear(256, 2)  ->  logits
```

| Aspect | Single-PTM | Multi-PTM | Multi-PTM conditioned |
|--------|-----------|-----------|----------------------|
| Training data | One PTM folder | All PTM folders pooled | All PTM folders pooled |
| Forward signature | `model(x)` | `model(x)` | `model(x, ptm_ids)` |
| Head input dim | 256 (GRU hidden) | 256 (GRU hidden) | 256 + 32 = 288 |
| PTM awareness | None needed (single task) | None (relies on feature patterns) | Learned 32-d embedding per PTM |
| Separate models needed | One per PTM (N models) | One model total | One model total |
| Class imbalance scope | Within one PTM | Across all PTMs combined | Across all PTMs combined |

### Why the PTM embedding matters

Without `--condition_on_ptm`, the model has no explicit signal about which PTM a sample belongs to. It must infer the modification type entirely from the Mamba hidden state patterns. This can work if PTM-Mamba already encodes enough PTM-specific information in its representations.

With `--condition_on_ptm`, the 32-dimensional PTM embedding tells the head which modification type it is classifying. This lets the shared CNN+GRU backbone learn general sequence features while the head adjusts its decision boundary per PTM. The embedding is learned end-to-end, so the model discovers whatever PTM-level bias is most useful for classification.

### When to prefer which

- **Single-PTM models** are the baseline. They are simpler, faster to train, and give a clean per-PTM benchmark.
- **Multi-PTM (unconditioned)** tests whether a single model generalizes across PTM types using shared patterns.
- **Multi-PTM (conditioned)** is the strongest variant. If it matches or slightly beats single-PTM baselines, it suggests that PTMs share learnable structure and benefit from joint training.

## Class Imbalance Handling

Two mechanisms address the varying imbalance across PTMs:

1. **Global class weights** in `CrossEntropyLoss` -- inverse frequency weighting across all pooled labels (positive vs negative).
2. **PTM-balanced sampling** (`--balance_ptm_batches`) -- each mini-batch draws from all PTMs proportionally, preventing large datasets from dominating training.
