#!/bin/bash
#SBATCH --job-name=multip_hparam
#SBATCH --account=protmamba
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/multip_hparam_%x_%j.out
#SBATCH --error=logs/multip_hparam_%x_%j.err

# ── configuration ─────────────────────────────────────────────────────────────
PTM_TYPES=${PTM_TYPES:-"acet_k,met_r,phos_y,sumo_k"}

FEATURES_ROOT="features"
OUTPUT_ROOT="classifier_runs"
BATCH_SIZE=64
NUM_EPOCHS=35
LR=1e-3
WEIGHT_DECAY=1e-2
VAL_SIZE=0.15
SEED=42
DEVICE="cuda:0"

# Search space (override from command line, e.g.
#   CONV_LAYERS_LIST="1 2" sbatch multip_ptm_classification/run_hparam_search.sbatch
MODEL_TYPES=${MODEL_TYPES:-"cnn_gru"}
CONV_LAYERS_LIST=${CONV_LAYERS_LIST:-"1 2 3"}
RNN_LAYERS_LIST=${RNN_LAYERS_LIST:-"1 2 3"}
LR_LIST=${LR_LIST:-"1e-3 5e-4"}
PTM_EMBED_DIM_LIST=${PTM_EMBED_DIM_LIST:-"32"}
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p logs

echo "========================================="
echo "Job ID            : $SLURM_JOB_ID"
echo "Node              : $SLURMD_NODENAME"
echo "PTM types         : $PTM_TYPES"
echo "Model types       : $MODEL_TYPES"
echo "Conv layers list  : $CONV_LAYERS_LIST"
echo "RNN layers list   : $RNN_LAYERS_LIST"
echo "LR list           : $LR_LIST"
echo "PTM embed dim list: $PTM_EMBED_DIM_LIST"
echo "Started at        : $(date)"
echo "========================================="

spack load /gfiltc4

apptainer exec --nv \
    --bind "$HOME/ptm-mamba:/ptm-mamba" \
    "$HOME/ptm-mamba/apptainer/pytorch_23.12.sif" \
    bash -c "
        source \$HOME/plm_native/bin/activate && \
        source \$HOME/.cargo/env && \
        cd /ptm-mamba && \
        python -m multip_ptm_classification.scripts.hparam_search \
            --ptm_types       ${PTM_TYPES} \
            --features_root   ${FEATURES_ROOT} \
            --output_root     ${OUTPUT_ROOT} \
            --batch_size      ${BATCH_SIZE} \
            --num_epochs      ${NUM_EPOCHS} \
            --lr              ${LR} \
            --weight_decay    ${WEIGHT_DECAY} \
            --val_size        ${VAL_SIZE} \
            --seed            ${SEED} \
            --device          ${DEVICE} \
            --condition_on_ptm \
            --balance_ptm_batches \
            --amp \
            --num_workers     2 \
            --prefetch_factor 2 \
            --model_types     ${MODEL_TYPES} \
            --conv_layers_list ${CONV_LAYERS_LIST} \
            --rnn_layers_list  ${RNN_LAYERS_LIST} \
            --lr_list          ${LR_LIST} \
            --ptm_embed_dim_list ${PTM_EMBED_DIM_LIST}
    "

echo "========================================="
echo "Finished at : $(date)"
echo "========================================="
