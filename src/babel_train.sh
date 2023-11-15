#!/bin/bash
#SBATCH --job-name=xlmr-defaults_training
#SBATCH --output xlmr_train.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48GB
#SBATCH --time 0-03:00:00
#SBATCH --partition=babel-shared-long

echo $SLURM_JOB_ID

source ~/.bashrc
conda init bash
conda activate figlang
python /home/shailyjb/anlp-figlang/src/Trainer.py --output_dir="/data/tir/projects/tir5/users/shailyjb/anlp-figlang/xlmr_large_defaults_with_logging" --data_dir="/home/shailyjb/anlp-figlang/data"