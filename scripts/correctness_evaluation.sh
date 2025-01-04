#!/bin/bash -l

 
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

### Specify Memory allocate to this job ###
#SBATCH --mem=64G

### Specify number of core (CPU) to allocate to per node ###
#SBATCH --ntasks-per-node=1

### Specify number of node to compute ###
#SBATCH --nodes=1

### Optional: Specify node to execute the job ###
### Remove 1st # at next line for the option to take effect ###
##SBATCH --nodelist=TC1N07

### Specify Time Limit, format: <min> or <min>:<sec> or <hr>:<min>:<sec> or <days>-<hr>:<min>:<sec> or <days>-<hr> ### 
#SBATCH --time=360

### Specify name for the job, filename format for output and error ###
#SBATCH --job-name=NeoCorr
#SBATCH --output=neocorr.out
#SBATCH --error=neocorr.err




conda activate creativity

BASE_PATH="/home/FYP/mohor001/NeoCoder"

cd $BASE_PATH
export PYTHONPATH=${BASE_PATH}


python3 steps/creativity_evaluation.py \
     --task correctness \
     --inference-result-path datasets/CodeForce/inference/Llama-3.1-8B-Instruct_sample=199_dp=5.json \
     --test-case-path datasets/CodeForce/NeoCoder/test_cases_annotated.json \
     --save-folder datasets/CodeForce/evaluation
