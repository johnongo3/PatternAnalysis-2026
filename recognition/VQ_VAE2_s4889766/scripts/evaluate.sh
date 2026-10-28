#!/bin/bash
# Evaluate one stage-1 run with predict.py all: reconstruct, sample, audit, fid and
# interpolate. The first argument is the run directory and the rest go to
# predict.py. Submit from the project folder once the run's training has finished:
#
#     sbatch -J eval_L2 jobs/evaluate.sh checkpoints/vqvae2_L2_ssim0
#     sbatch -J eval_L2 jobs/evaluate.sh checkpoints/vqvae2_L2_ssim0 \
#         --temperatures 0.9 1.0 1.1 --bottom-temperatures 0.5 0.6 0.7
#
#SBATCH --job-name=vqvae2_eval
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out

source $HOME/miniconda3/bin/activate
conda activate vqvae2
set -eo pipefail

RUN=${1:?usage: sbatch jobs/evaluate.sh RUN [predict.py options]}
shift
python predict.py all "$RUN" "$@"
