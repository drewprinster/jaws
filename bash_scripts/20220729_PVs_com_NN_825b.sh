#!/bin/sh
#SBATCH -t 4- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=20 
#SBATCH --mem=100G 
python3 ../run_JAW_PredictiveValuesForAUC.py --dataset communities --muh_fun_name NN --bias 0.825 --ntrial 10
