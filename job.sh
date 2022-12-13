#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

module purge
module load Python/3.6.4-foss-2018a
#module load MATLAB/2020b
module load MATLAB-Engine/2019b-GCCcore-8.3.0
#module load cuDNN/7.4.2.24-CUDA-10.0.130
module load cuDNN/7.6.4.38-gcccuda-2019b

source /data/$USER/.envs/acf/bin/activate

python3 ./main.py --model densenet121 --method 5_fold --dataset_rgb '~/data/extractions/line-centre-gray/' --resize 224
 
deactivate
