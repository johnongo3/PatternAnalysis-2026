#!/bin/bash
# Train one model end to end on Rangpur: stage 1, then its priors. Any arguments
# are passed to every train.py call. Submit from the project folder:
#
#     sbatch jobs/train.sh                          # VQ-VAE-2
#     sbatch jobs/train.sh --levels 1               # single-level baseline
#     sbatch jobs/train.sh --ssim-loss-weight 0.01  # SSIM-loss variant
#
#SBATCH --job-name=vqvae2_train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out

source $HOME/miniconda3/bin/activate
conda activate vqvae2
set -eo pipefail

python train.py "$@"
case " $* " in
    *" --levels 1 "* | *" --levels=1 "*) ;; # the baseline has no top level
    *) python train.py --stage top "$@" ;;
esac
python train.py --stage bottom "$@"
