#!/bin/bash
#SBATCH --job-name=xlmr-su-inf
#SBATCH --output=xlmr_su-inf.out
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

cp -r /data/tir/projects/tir5/users/shailyjb/anlp-figlang/su_50/checkpoint-960 /home/shailyjb/anlp-figlang/few_shot_trained_models/xlmr_su_50
echo "Copied checkpoint"

python /home/shailyjb/anlp-figlang/src/Inference.py --ckpt_path="/home/shailyjb/anlp-figlang/few_shot_trained_models/xlmr_su_50" --data_dir="/home/shailyjb/anlp-figlang/data/few_shot" --split="test_50" --output_dir="/home/shailyjb/anlp-figlang/few_shot_50_out/xlmr_su"
echo "Done"