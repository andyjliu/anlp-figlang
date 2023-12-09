#!/bin/bash
#SBATCH --job-name=mt5_small_trial_2_epochs
#SBATCH --output=anlp-figlang/mt5_small_trial_2_epochs.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48GB
#SBATCH --time 0-03:00:00
#SBATCH --partition=babel-shared
#SBATCH --mail-type=END
#SBATCH --mail-user=shailyjb@andrew.cmu.edu

echo $SLURM_JOB_ID

source ~/.bashrc
conda init bash
conda activate figlang
python /home/shailyjb/anlp-figlang/src/mT5FineTuning.py \
    --output_dir="/data/tir/projects/tir5/users/shailyjb/anlp-figlang/mt5_small_trial_2_epochs" \
    --train_file "/home/shailyjb/anlp-figlang/data/train/en.csv" \
    --val_file "/home/shailyjb/anlp-figlang/data/validation/en.csv"