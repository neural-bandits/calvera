#!/bin/bash
#SBATCH -A herbrich-student
#SBATCH --partition cauldron
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=64GB
#SBATCH --gpus=1
#SBATCH --output=experiments/logs/log_%j.log
#SBATCH --constraint=ARCH:X86

#SBATCH --job-name=calvera
#SBATCH --time=24:00:00

#source ~/.bashrc # need this to allow activating a conda env
#conda activate calvera

python src/calvera/benchmark/benchmark.py experiments/$1/config.yaml
