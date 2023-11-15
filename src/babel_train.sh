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
python /home/shailyjb/anlp-figlang/src/Trainer.py --output_dir="/data/tir/projects/tir5/users/shailyjb/anlp-figlang/xlmr_large_defaults_with_logging" --train-file "/home/shailyjb/anlp-figlang/data/train/en.csv" --validation-file "/home/shailyjb/anlp-figlang/data/validation/en.csv"
python /home/shailyjb/anlp-figlang/src/Trainer.py --output_dir="/data/tir/projects/tir5/users/shailyjb/anlp-figlang/xlmr_large_defaults_with_logging" --train-file "/home/shailyjb/anlp-figlang/data/few_shot/train_merged/hi/hi_50.csv" --validation-file "/home/shailyjb/anlp-figlang/data/validation/en.csv"
python /home/shailyjb/anlp-figlang/src/Trainer.py --output_dir="/data/tir/projects/tir5/users/shailyjb/anlp-figlang/xlmr_large_defaults_with_logging" --train-file "/home/shailyjb/anlp-figlang/data/few_shot/train_merged/id/id_50.csv" --validation-file "/home/shailyjb/anlp-figlang/data/validation/en.csv"
python /home/shailyjb/anlp-figlang/src/Trainer.py --output_dir="/data/tir/projects/tir5/users/shailyjb/anlp-figlang/xlmr_large_defaults_with_logging" --train-file "/home/shailyjb/anlp-figlang/data/few_shot/train_merged/jv/jv_50.csv" --validation-file "/home/shailyjb/anlp-figlang/data/validation/en.csv"
python /home/shailyjb/anlp-figlang/src/Trainer.py --output_dir="/data/tir/projects/tir5/users/shailyjb/anlp-figlang/xlmr_large_defaults_with_logging" --train-file "/home/shailyjb/anlp-figlang/data/few_shot/train_merged/kn/kn_50.csv" --validation-file "/home/shailyjb/anlp-figlang/data/validation/en.csv"
python /home/shailyjb/anlp-figlang/src/Trainer.py --output_dir="/data/tir/projects/tir5/users/shailyjb/anlp-figlang/xlmr_large_defaults_with_logging" --train-file "/home/shailyjb/anlp-figlang/data/few_shot/train_merged/su/su_50.csv" --validation-file "/home/shailyjb/anlp-figlang/data/validation/en.csv"
python /home/shailyjb/anlp-figlang/src/Trainer.py --output_dir="/data/tir/projects/tir5/users/shailyjb/anlp-figlang/xlmr_large_defaults_with_logging" --train-file "/home/shailyjb/anlp-figlang/data/few_shot/train_merged/sw/sw_50.csv" --validation-file "/home/shailyjb/anlp-figlang/data/validation/en.csv"
python /home/shailyjb/anlp-figlang/src/Trainer.py --output_dir="/data/tir/projects/tir5/users/shailyjb/anlp-figlang/xlmr_large_defaults_with_logging" --train-file "/home/shailyjb/anlp-figlang/data/few_shot/train_merged/yo/yo_50.csv" --validation-file "/home/shailyjb/anlp-figlang/data/validation/en.csv"