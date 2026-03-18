#!/bin/bash
#SBATCH --job-name=ptm_infer
#SBATCH --account=protmamba
#SBATCH --partition=interactive
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=clx-a-01
#SBATCH --time=6:00:00
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err

mkdir -p logs

# Load apptainer
spack load /gfiltc4

# Run inference inside the container
# - --nv passes through GPU drivers
# - --bind mounts the project directory
# - source the venv and cargo env, then run the script
apptainer exec --nv \
    --bind "$HOME/ptm-mamba:/ptm-mamba" \
    "$HOME/ptm-mamba/apptainer/pytorch_23.12.sif" \
    bash -c "
        source \$HOME/plm_native/bin/activate && \
        source \$HOME/.cargo/env && \
        cd /ptm-mamba && \
        python protein_lm/modeling/scripts/infer.py
    "
