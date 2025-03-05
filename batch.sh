#!/bin/bash
#SBATCH -A herbrich-student
#SBATCH --partition magic
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --output=experiments/logs/log_%j.log
#SBATCH --constraint=ARCH:X86

#SBATCH --job-name=calvera
#SBATCH --time=24:00:00

#source ~/.bashrc # need this to allow activating a conda env
#conda activate calvera

python src/calvera/benchmark/benchmark.py experiments/$1/config.yaml
