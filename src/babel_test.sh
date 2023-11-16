#!/bin/bash
#SBATCH --job-name=xlmr_zs_en_dev
#SBATCH --output=xlmr_zs_en_dev.out
#SBATCH --nodes=1
#SBATCH --mem=48GB
#SBATCH --time 0-03:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=babel-shared-long
#SBATCH --mail-type=END
#SBATCH --mail-user=shailyjb@andrew.cmu.edu

echo $SLURM_JOB_ID

source ~/.bashrc
conda init bash
conda activate figlang

# cp -r /data/tir/projects/tir5/users/shailyjb/anlp-figlang/su_50/checkpoint-960 /home/shailyjb/anlp-figlang/few_shot_trained_models/xlmr_su_50
# echo "Copied checkpoint"

python /home/shailyjb/anlp-figlang/src/Inference.py --ckpt_path="/home/shailyjb/anlp-figlang/xlmr_large_end_ckpt" --data_dir="/home/shailyjb/anlp-figlang/data" --split="validation" --output_dir="/home/shailyjb/anlp-figlang/xlmr_large_end_ckpt/test_out"
echo "Done"