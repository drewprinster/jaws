#!/bin/sh
#SBATCH -t 1- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 
#SBATCH --mem=80G
python3 ../run_JAWA_expts.py --dataset wave --bias 0.0000925 --ntrial 10 --L2_lambda 4.0