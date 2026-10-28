#!/bin/bash
# Compare the evaluated runs: the failure autopsy at the recommended setting,
# resource use, the results tables (assets/results.md) and the comparison plots.
# Submit from the project folder once every evaluation job has finished:
#
#     sbatch jobs/report.sh
#SBATCH --job-name=vqvae2_report
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out

source $HOME/miniconda3/bin/activate
conda activate vqvae2
set -eo pipefail

VQVAE2=checkpoints/vqvae2_L2_ssim0
BASELINE=checkpoints/vqvae2_L1_ssim0
SSIM=checkpoints/vqvae2_L2_ssim0.01

python predict.py autopsy $SSIM --setting 1.1/0.5
python predict.py resources $VQVAE2 $BASELINE
python predict.py report $VQVAE2 $BASELINE $SSIM
python scripts/plot_results.py assets/results.md
