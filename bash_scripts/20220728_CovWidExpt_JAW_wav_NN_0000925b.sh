#!/bin/sh
#SBATCH -t 2- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 
#SBATCH --mem=50G 
python3 ../run_JAW_expts.py --dataset wave --muh_fun_name NN --bias 0.0000925 --ntrial 10