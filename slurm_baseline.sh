#!/bin/bash

#SBATCH --job-name=Baseline
#SBATCH --partition=milan,genoa
#SBATCH --exclude=mg15,g[09-12]  # Exclude: mg15 (critical issues), g09-12 (L4)
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2%2  # 7 models × 3 datasets = 21 tasks, max 5 concurrent jobs
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null


source /nesi/project/uoo04109/miniforge3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1 # don't add python user site library to path
conda activate torch

# Support both A100 (8.0) and H100 (9.0)
export TORCH_CUDA_ARCH_LIST="8.0,9.0"

cd /nesi/project/uoo04109/practice/Baseline_RecBole
mkdir -p logs_baseline/

# Define model and dataset arrays
# models=("SASRec" "FEARec" "SINE" "CORE" "SASRecCPR" "BERT4Rec" "DuoRec" "CL4SRec" "TedRec")
models=("DuoRec" "CL4SRec")
datasets=("Industrial_and_Scientific" "Baby_Products" "Office_Products")
model_idx=$((SLURM_ARRAY_TASK_ID / ${#datasets[@]}))
dataset_idx=$((SLURM_ARRAY_TASK_ID % ${#datasets[@]}))
model=${models[$model_idx]}
dataset=${datasets[$dataset_idx]}

# Redirect output to custom named files
exec 1> "logs_baseline/${model}_${dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
exec 2> "logs_baseline/${model}_${dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

echo "Running job with Job-ID: $SLURM_JOB_ID, Array Job ID: $SLURM_ARRAY_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "========================================="
echo "Experiment started at: $(date)"
echo "Model: $model"
echo "Dataset: $dataset"
echo "========================================="

nvidia-smi

python main.py --model="$model" --dataset="$dataset" --log_wandb

echo "========================================="
echo "Experiment completed at: $(date)"
echo "========================================="