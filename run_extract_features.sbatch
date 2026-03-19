#!/bin/bash
#SBATCH --job-name=ptm_extract2
#SBATCH --account=protmamba
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=8g
#SBATCH --output=logs/extract_%x_%j.out
#SBATCH --error=logs/extract_%x_%j.err

# ── configuration ─────────────────────────────────────────────────────────────
# Change PTM_TYPE to switch to a different PTM (e.g. phospho_s, ubiq_k, etc.)
PTM_TYPE=${PTM_TYPE:-met_r}

CKPT_PATH="ckpt/best.ckpt"
OUT_ROOT="features"
BATCH_SIZE=64
DEVICE="cuda:0"
DATA_ROOT="datasets/$PTM_TYPE"
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p logs

echo "========================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "PTM type    : $PTM_TYPE"
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
        python protein_lm/modeling/scripts/extract_features.py \
            --ptm_type   ${PTM_TYPE} \
            --ckpt_path  ${CKPT_PATH} \
            --out_root   ${OUT_ROOT} \
            --batch_size ${BATCH_SIZE} \
            --device     ${DEVICE} \
            --data_root  ${DATA_ROOT}
    "

echo "========================================="
echo "Finished at : $(date)"
echo "========================================="
