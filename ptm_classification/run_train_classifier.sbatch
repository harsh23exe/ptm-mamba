#!/bin/bash
#SBATCH --job-name=ptm_train
#SBATCH --account=protmamba
#SBATCH --partition=interactive
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=clx-a-01
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err

# ── configuration ─────────────────────────────────────────────────────────────
# Change PTM_TYPE to switch PTM (e.g. phospho_s, ubiq_k, etc.)
PTM_TYPE=${PTM_TYPE:-acet_k}

# Model type: cnn_gru or cnn_bilstm
# Override at submit time, e.g.:
#   MODEL_TYPE=cnn_bilstm sbatch ptm_classification/run_train_classifier.sbatch
MODEL_TYPE=${MODEL_TYPE:-cnn_gru}

FEATURES_ROOT="features"
OUTPUT_ROOT="classifier_runs"
BATCH_SIZE=64
NUM_EPOCHS=20
LR=1e-3
WEIGHT_DECAY=1e-2
VAL_SIZE=0.15
SEED=42
DEVICE="cuda:0"
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p logs

echo "========================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "PTM type    : $PTM_TYPE"
echo "Model type  : $MODEL_TYPE"
echo "Started at  : $(date)"
echo "========================================="

spack load /gfiltc4

apptainer exec --nv \
    --bind "$HOME/ptm-mamba:/ptm-mamba" \
    "$HOME/ptm-mamba/apptainer/pytorch_23.12.sif" \
    bash -c "
        source \$HOME/plm_native/bin/activate && \
        source \$HOME/.cargo/env && \
        cd /ptm-mamba && \
        python -m ptm_classification.scripts.train_classifier \
            --ptm_type      ${PTM_TYPE} \
            --features_root ${FEATURES_ROOT} \
            --output_root   ${OUTPUT_ROOT} \
            --model_type    ${MODEL_TYPE} \
            --batch_size    ${BATCH_SIZE} \
            --num_epochs    ${NUM_EPOCHS} \
            --lr            ${LR} \
            --weight_decay  ${WEIGHT_DECAY} \
            --val_size      ${VAL_SIZE} \
            --seed          ${SEED} \
            --device        ${DEVICE}
    "

echo "========================================="
echo "Finished at : $(date)"
echo "========================================="

