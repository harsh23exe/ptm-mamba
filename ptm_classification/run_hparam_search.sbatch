#!/bin/bash
#SBATCH --job-name=ptm_hparam
#SBATCH --account=protmamba
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/hparam_%x_%j.out
#SBATCH --error=logs/hparam_%x_%j.err
#SBATCH --mem=32g
#SBATCH --cpus-per-task=8

# ── configuration ─────────────────────────────────────────────────────────────
# Single PTM type per run (matches the original acet_k workflow).
PTM_TYPE=${PTM_TYPE:-crot_k}

FEATURES_ROOT="features"
OUTPUT_ROOT="classifier_runs"
BATCH_SIZE=64
NUM_EPOCHS=35
LR=1e-3
WEIGHT_DECAY=1e-2
VAL_SIZE=0.15
SEED=42
DEVICE="cuda:0"

# Model / architecture search space
# Override from command line if desired, e.g.:
#   MODEL_TYPES="cnn_bilstm" sbatch ptm_classification/run_hparam_search.sbatch
MODEL_TYPES=${MODEL_TYPES:-"cnn_gru cnn_bilstm"}
CONV_LAYERS_LIST=${CONV_LAYERS_LIST:-"1 2 3"}
RNN_LAYERS_LIST=${RNN_LAYERS_LIST:-"1 2 3"}
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p logs

echo "========================================="
echo "Job ID          : $SLURM_JOB_ID"
echo "Node            : $SLURMD_NODENAME"
echo "PTM type        : $PTM_TYPE"
echo "Model types     : $MODEL_TYPES"
echo "Conv layers list: $CONV_LAYERS_LIST"
echo "RNN layers list : $RNN_LAYERS_LIST"
echo "Started at      : $(date)"
echo "========================================="

spack load /gfiltc4

apptainer exec --nv \
    --bind "$HOME/ptm-mamba:/ptm-mamba" \
    "$HOME/ptm-mamba/apptainer/pytorch_23.12.sif" \
    bash -c "
        source \$HOME/plm_native/bin/activate && \
        source \$HOME/.cargo/env && \
        cd /ptm-mamba && \
        echo \"===== Starting hyperparameter search for PTM_TYPE=${PTM_TYPE} =====\" && \
        python -m ptm_classification.scripts.hparam_search \
            --ptm_type      ${PTM_TYPE} \
            --features_root ${FEATURES_ROOT} \
            --output_root   ${OUTPUT_ROOT} \
            --batch_size    ${BATCH_SIZE} \
            --num_epochs    ${NUM_EPOCHS} \
            --lr            ${LR} \
            --weight_decay  ${WEIGHT_DECAY} \
            --val_size      ${VAL_SIZE} \
            --seed          ${SEED} \
            --device        ${DEVICE} \
            --model_types   ${MODEL_TYPES} \
            --conv_layers_list ${CONV_LAYERS_LIST} \
            --rnn_layers_list  ${RNN_LAYERS_LIST}
    "

echo "========================================="
echo "Finished at : $(date)"
echo "========================================="

