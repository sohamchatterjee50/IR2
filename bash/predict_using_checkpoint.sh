#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --job-name=IR2
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem=32000M
#SBATCH --output=Setup1.out
pip cache purge
#module avail


#module avail Python


# module purge 
# module load 2021
# module avail Python
# # module load Python/3.9.5-GCCcore-10.3.0

# module load Python/2.7.18-GCCcore-10.3.0-bare
# module purge
# module load 2022
# module load Anaconda3/2022.05


module purge
module load 2023
module load Anaconda3/2023.07-2

# Run your code
conda info --envs


cd /home/scur2840/IR2-stefan/IR2

source activate env_ir5

python retriever_predict.py