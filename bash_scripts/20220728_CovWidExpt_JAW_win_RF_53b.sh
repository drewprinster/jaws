#!/bin/sh
#SBATCH -t 2- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=50G 
python3 ../run_JAW_expts.py --dataset wine --muh_fun_name RF --bias 0.53 --ntrial 10
