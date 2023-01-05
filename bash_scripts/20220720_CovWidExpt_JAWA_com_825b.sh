#!/bin/sh
#SBATCH -t 1- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=20 
#SBATCH --mem=100G
python3 ../run_JAWA_expts.py --dataset communities --bias 0.825 --ntrial 10 --L2_lambda 64.0