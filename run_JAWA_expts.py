## File last updated January 4, 2023

import imp
import logging
imp.reload(logging)
logging.basicConfig(level=logging.INFO)

from functools import partial

import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
import stan
import seaborn as sns

from scipy.optimize import minimize

from utils import bayesnn

import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy
import math
import tqdm
from datetime import datetime
from datetime import date
from random import sample

import random
from sklearn import decomposition
from datetime import date
import argparse
import warnings

import sys,os
sys.path.insert(0,os.getcwd() + '../')


from utils.JAWS_utils import *



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run JAW experiments with given bias, # trials, mu func, and dataset.')
    
    parser.add_argument('--dataset', type=str, default='airfoil', help='Dataset for experiments.')
    parser.add_argument('--bias', type=float, default=1.0, help='Scalar bias magnitude parameter for exponential tilting covariate shift.')
    parser.add_argument('--ntrial', type=int, default=10, help='Number of trials (experiment replicates) to complete.')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha value corresponding to 1-alpha target coverage')
    parser.add_argument('--ntrain', type=int, default=200, help='Number of training datapoints')
    parser.add_argument('--L2_lambda', type=float, default=1.0, help='L2 lamdba')
    parser.add_argument('--grid_search', type=bool, default=False, help='Do grid search for IF Hessian dampening parameter?')

    # python 20220716_run_JAWA.py dataset muh_fun ntrial bias alpha
    # python 20220716_run_JAWA.py airfoil RF 20 1 0.1
    

    args = parser.parse_args()
    dataset = args.dataset
    bias = args.bias
    ntrial = args.ntrial
    alpha = args.alpha
    n = args.ntrain
    L2_lambda = args.L2_lambda
    grid_search = args.grid_search
    
    warnings.simplefilter(action='ignore', category=FutureWarning)

    
    if (dataset == 'airfoil'):
        airfoil = pd.read_csv(os.getcwd().removesuffix('bash_scripts') + 'datasets/airfoil/airfoil.txt', sep = '\t', header=None)
        airfoil.columns = ["Frequency","Angle","Chord","Velocity","Suction","Sound"]
        X_airfoil = airfoil.iloc[:, 0:5].values
        X_airfoil[:, 0] = np.log(X_airfoil[:, 0])
        X_airfoil[:, 4] = np.log(X_airfoil[:, 4])
        Y_airfoil = airfoil.iloc[:, 5].values
        n_airfoil = len(Y_airfoil)
        print("X_airfoil shape : ", X_airfoil.shape)
        
    elif (dataset == 'wine'):
        winequality_red = pd.read_csv(os.getcwd().removesuffix('bash_scripts') + 'datasets/wine/winequality-red.csv', sep=';')
        X_wine = winequality_red.iloc[:, 0:11].values
        Y_wine = winequality_red.iloc[:, 11].values
        n_wine = len(Y_wine)
        print("X_wine shape : ", X_wine.shape)
        
    elif (dataset == 'wave'):
        wave = pd.read_csv(os.getcwd().removesuffix('bash_scripts') + 'datasets/WECs_DataSet/Adelaide_Data.csv', header = None)
        X_wave = wave.iloc[0:2000, 0:48].values
        Y_wave = wave.iloc[0:2000, 48].values
        n_wave = len(Y_wave)
        print("X_wave shape : ", X_wave.shape)
        
    elif (dataset == 'superconduct'):
        superconduct = pd.read_csv(os.getcwd().removesuffix('bash_scripts') + 'datasets/superconduct/train.csv')
        X_superconduct = superconduct.iloc[0:2000, 0:81].values
        Y_superconduct = superconduct.iloc[0:2000, 81].values
        n_superconduct = len(Y_superconduct)
        print("X_superconduct shape : ", X_superconduct.shape)
        
    elif (dataset == 'communities'):
        # UCI Communities and Crime Data Set
        # download from:
        # http://archive.ics.uci.edu/ml/datasets/communities+and+crime
        communities_data = np.loadtxt(os.getcwd().removesuffix('bash_scripts') + 'datasets/communities/communities.data',delimiter=',',dtype=str)
        # remove categorical predictors
        communities_data = np.delete(communities_data,np.arange(5),1)
        # remove predictors with missing values
        communities_data = np.delete(communities_data,\
                    np.argwhere((communities_data=='?').sum(0)>0).reshape(-1),1)
        communities_data = communities_data.astype(float)
        X_communities = communities_data[:,:-1]
        Y_communities = communities_data[:,-1]
        n_communities = len(Y_communities)
        print("X_communities shape : ", X_communities.shape)
        
    else:
        raise Exception("Invalid dataset name")
        
    if (n >= eval('n_'+dataset)):
        raise Exception("Error: number of training datapoints is greater than total number of datapoints")

    if (grid_search):
        np.random.RandomState(12345)

        val_size = 200
        
        val_inds = np.random.choice(eval('n_'+dataset), val_size, replace=False)

        pd.DataFrame(val_inds).to_csv(os.getcwd().removesuffix('bash_scripts') + 'IF_intermediate_files/val_inds_' + dataset + '.csv') ## record validation indices
            
        print("val_inds : ", val_inds)

        alpha = 0.1
        damp = 0.0 ## default dampening
        bias = 0 ## Bias = 0 for grid search
        method_names = ['IF1-jackknife', 'IF1-jackknife-mm', 'IF1-jackknife+', 'IF1-JAWA', 'IF1-JAWAmm',
                        'IF2-jackknife', 'IF2-jackknife-mm', 'IF2-jackknife+', 'IF2-JAWA', 'IF2-JAWAmm',
                        'IF3-jackknife', 'IF3-jackknife-mm', 'IF3-jackknife+', 'IF3-JAWA', 'IF3-JAWAmm']

        grid_values = np.array([0.5, 1, 2, 4, 8, 16, 32, 64, 96, 128])

        n = 200 ## Training size



        results = pd.DataFrame(columns = ['itrial','dataset','method','coverage','width', 'L2_lambda'])

        for itrial in range(ntrial):
            non_val_inds = list(np.setdiff1d(np.arange(eval('n_'+dataset)), val_inds))
            train_inds = random.sample(non_val_inds, n)

            X_dataset = eval('X_'+dataset)
            Y_dataset = eval('Y_'+dataset)

            X = X_dataset[train_inds, :]
            Y = Y_dataset[train_inds]
            X1 = X_dataset[val_inds, :]
            Y1 = Y_dataset[val_inds]

            X_full = np.concatenate((X, X1), axis = 0)

            weights_full = np.ones(len(X_full))

            ## Normalize data
            norm_X = InputNormalizer(X)
            norm_y = TargetNormalizer(Y)

            X = norm_X.normalize(X)
            Y = norm_y.normalize(Y)

            X1 = norm_X.normalize(X1)
            Y1 = norm_y.normalize(Y1)


            for L2_lambda in tqdm.tqdm(grid_values):
                print("dataset : ", dataset, "lambda_l2 = ", L2_lambda)

                PIs = compute_PIs_IFs(X,Y,X1,alpha, weights_full, dataset, bias, L2_lambda, itrial)
                for method in method_names:
                    coverage = ((PIs[method]['lower'] <= Y1)&(PIs[method]['upper'] >= Y1)).mean()
                    width = (PIs[method]['upper'] - PIs[method]['lower']).median()
                    results.loc[len(results)]=\
                    [itrial,dataset,method,coverage,width,L2_lambda]

        results.to_csv(os.getcwd().removesuffix('bash_scripts') + 'IF_intermediate_files/'+ str(date.today()) + '_IFs_L2GridSearch_' + dataset + '_' + str(ntrial) +'Trials.csv',index=False)

        
    else:
        print("Running main JAWA experiments")
        print("Dataset : ", dataset)
        print("bias = ", bias)
        print("L2_lambda = ", L2_lambda)
        
        val_inds = pd.read_csv(os.getcwd().removesuffix('bash_scripts') + 'IF_intermediate_files/val_inds_' + dataset + '.csv', index_col = 0).T.to_numpy()[0]
        print("val_ids : ", val_inds)

        np.random.RandomState(12345)

        alpha = 0.1
        damp = 0.0 ## default dampening
        method_names = ['IF1-jackknife', 'IF1-jackknife-mm', 'IF1-jackknife+', 'IF1-JAWA', 'IF1-JAWA', 
                        'IF2-jackknife', 'IF2-jackknife-mm', 'IF2-jackknife+', 'IF2-JAWA', 'IF2-JAWAmm',
                        'IF3-jackknife', 'IF3-jackknife-mm', 'IF3-jackknife+', 'IF3-JAWA', 'IF3-JAWAmm',]

        n = 200 ## Training size

        results = pd.DataFrame(columns = ['itrial','dataset','method','coverage','width', 'L2_lambda'])

        for itrial in range(ntrial):
            non_val_inds = list(np.setdiff1d(np.arange(eval('n_'+dataset)), val_inds))
            train_inds = random.sample(non_val_inds, n)
            test_inds = np.setdiff1d(non_val_inds, train_inds)


            X_dataset = eval('X_'+dataset)
            Y_dataset = eval('Y_'+dataset)

            X = X_dataset[train_inds, :]
            Y = Y_dataset[train_inds]
            X1 = X_dataset[test_inds, :]
            Y1 = Y_dataset[test_inds]

            X1_unshifted = X1


            biased_test_indices = exponential_tilting_indices(X, X1, dataset, bias=bias)

            ## Bias the test data if bias != 0
            if (bias != 0):
                ## Test data exponential tilting indices

                X1 = X1[biased_test_indices] ## Apply shift to X1
                Y1 = Y1[biased_test_indices] ## Apply shift to Y1


            X_full = np.concatenate((X, X1), axis = 0)


            weights_full = get_w(X, X_full, dataset, bias).reshape(len(X_full)) ## Get weights for the full data

            ## Normalize data
            norm_X = InputNormalizer(X)
            norm_y = TargetNormalizer(Y)

            X = norm_X.normalize(X)
            Y = norm_y.normalize(Y)

            X1 = norm_X.normalize(X1)
            Y1 = norm_y.normalize(Y1)
            
            PIs = compute_PIs_IFs(X,Y,X1,alpha, weights_full, dataset, bias, L2_lambda, itrial, compute_PDs=False)
            
#             PIs = compute_PIs(X,Y,X1,alpha,muh_fun, weights_full, dataset, bias, run_effective_sample_size)
            for method in method_names:
                coverage = ((PIs[method]['lower'] <= Y1)&(PIs[method]['upper'] >= Y1)).mean()
                width = (PIs[method]['upper'] - PIs[method]['lower']).median()
                results.loc[len(results)]=\
                [itrial,dataset,method,coverage,width,L2_lambda]
                
       
        results.to_csv(os.getcwd().removesuffix('bash_scripts') + 'results/JAWA_coverage_width/'+ str(date.today()) + '_IFs_Experiments_' + dataset + '_' + str(ntrial) +'Trials_final.csv',index=False)
