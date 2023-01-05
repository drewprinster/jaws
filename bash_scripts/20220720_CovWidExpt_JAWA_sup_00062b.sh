#!/bin/sh
#SBATCH -t 1- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=20 
#SBATCH --mem=100G
python3 ../run_JAWA_expts.py --dataset superconduct --bias 0.00062 --ntrial 10 --L2_lambda 96.0