#!/bin/bash
#SBATCH --job-name=multip_train
#SBATCH --account=protmamba
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:2
#SBATCH --time=24:00:00
#SBATCH --mem=32g
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/multip_train_%x_%j.out
#SBATCH --error=logs/multip_train_%x_%j.err

# ── configuration ─────────────────────────────────────────────────────────────
PTM_TYPES=${PTM_TYPES:-"acet_k,met_r,phos_y,sumo_k"}
MODEL_TYPE=${MODEL_TYPE:-"cnn_gru"}

FEATURES_ROOT="features"
OUTPUT_ROOT="classifier_runs"
BATCH_SIZE=256
GRAD_ACCUM=2
NUM_EPOCHS=30
LR=1e-3
WEIGHT_DECAY=1e-2
VAL_SIZE=0.15
SEED=42
DEVICE="cuda:0"
NUM_WORKERS=8
PREFETCH=4
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p logs

echo "========================================="
echo "Job ID          : $SLURM_JOB_ID"
echo "Node            : $SLURMD_NODENAME"
echo "PTM types       : $PTM_TYPES"
echo "Model type      : $MODEL_TYPE"
echo "Batch size      : $BATCH_SIZE"
echo "Grad accum      : $GRAD_ACCUM"
echo "GPUs            : 2 x A100"
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
        python -m multip_ptm_classification.scripts.train_multip \
            --ptm_types       ${PTM_TYPES} \
            --features_root   ${FEATURES_ROOT} \
            --output_root     ${OUTPUT_ROOT} \
            --model_type      ${MODEL_TYPE} \
            --condition_on_ptm \
            --balance_ptm_batches \
            --batch_size      ${BATCH_SIZE} \
            --num_epochs      ${NUM_EPOCHS} \
            --lr              ${LR} \
            --weight_decay    ${WEIGHT_DECAY} \
            --val_size        ${VAL_SIZE} \
            --seed            ${SEED} \
            --device          ${DEVICE} \
            --amp \
            --data_parallel \
            --grad_accum_steps ${GRAD_ACCUM} \
            --num_workers     ${NUM_WORKERS} \
            --prefetch_factor ${PREFETCH}
    "

echo "========================================="
echo "Finished at : $(date)"
echo "========================================="
