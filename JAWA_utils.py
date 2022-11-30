import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
np.random.seed(98765)

## Drew added
import tqdm as tqdm
import random
from sklearn import decomposition

# positive_infinity = np.array([float('inf')])

def sort_both_by_first(v, w):
    zipped_lists = zip(v, w)
    sorted_zipped_lists = sorted(zipped_lists)
    v_sorted = [element for element, _ in sorted_zipped_lists]
    w_sorted = [element for _, element in sorted_zipped_lists]
    
    return [v_sorted, w_sorted]
    

def weighted_quantile(v, w_normalized, q):
    if (len(v) != len(w_normalized)):
        raise ValueError('Error: v is length ' + str(len(v)) + ', but w_normalized is length ' + str(len(w_normalized)))
        
    if (np.sum(w_normalized) > 1.01 or np.sum(w_normalized) < 0.99):
        raise ValueError('Error: w_normalized does not add to 1')
        
    if (q < 0 or 1 < q):
        raise ValueError('Error: Invalid q')

    n = len(v)
    
    v_sorted, w_sorted = sort_both_by_first(v, w_normalized)
    
    cum_w_sum = w_sorted[0]
    i = 0
    while(cum_w_sum <= q):
            i += 1
            cum_w_sum += w_sorted[i]
            
    if (q > 0.5): ## If taking upper quantile: ceil
        return v_sorted[i]
            
    elif (q < 0.5): ## Elif taking lower quantile:
        if (i > 0):
            return v_sorted[i-1]
        else:
            return v_sorted[0]
        
    else: ## Else taking median, return weighted average if don't have cum_w_sum == 0.5
        if (cum_w_sum == 0.5):
            return v_sorted[i]
        
        elif (i > 0):
            return (v_sorted[i]*w_sorted[i] + v_sorted[i-1]*w_sorted[i-1]) / (w_sorted[i] + w_sorted[i-1])
        
        else:
            return v_sorted[0]
        

def get_w(x_pca, x, dataset, bias):
    if (dataset=='airfoil'):
        return np.exp(x[:,[0,4]] @ [-bias,bias])
    
    elif(dataset == 'wine'):
        return np.exp(x[:,[0,10]] @ [-bias,bias])
    
    
    ## For communities dataset use top 2 PCs as tilting vars
    elif (dataset == 'communities'):
        np.random.seed(5)
        pca = decomposition.PCA(n_components=2)
        pca.fit(x_pca)
        x_red = pca.transform(x)
        return np.exp(x_red @ [-bias,bias])
    
    ## For communities dataset use top 2 PCs as tilting vars
    elif (dataset == 'superconduct'):
        np.random.seed(5)
        pca = decomposition.PCA(n_components=1)
        pca.fit(x_pca)
        x_red = pca.transform(x)
        return np.exp(x_red @ [bias])
    
    ## For blog and meps data use logarithm of top 2 PCs as tilting vars
    else: 
        np.random.seed(5)
        pca = decomposition.PCA(n_components=2)
        pca.fit(x_pca)
        x_red = pca.transform(x)
        x_red_min = abs(x_red.min(axis=0))
        x_red_adjusted = x_red + np.tile(x_red_min + 1, (len(x_red), 1))
        log_x_red_adjusted = np.log(x_red_adjusted)
        return np.exp(log_x_red_adjusted @ [-bias,bias])
    
    
def get_w_normalized(x_pca, x, dataset, bias):
    if (dataset=='airfoil'):
        return np.exp(x[:,[0,4]] @ [-bias,bias])
    
    elif(dataset == 'wine'):
        return np.exp(x[:,[0,10]] @ [-bias,bias])
    
    elif(dataset == 'concrete'):
        return np.exp(np.log(x[:,[0,6]]) @ [-bias,bias])
    
    ## For communities dataset use top 2 PCs as tilting vars
    elif (dataset in ['communities', 'naval']):
        np.random.seed(5)
        pca = decomposition.PCA(n_components=2)
        pca.fit(x_pca)
        x_red = pca.transform(x)
        return np.exp(x_red @ [-bias,bias])
    
    ## For communities dataset use top 2 PCs as tilting vars
    elif (dataset in ['superconduct']):
        np.random.seed(5)
        pca = decomposition.PCA(n_components=1)
        pca.fit(x_pca)
        x_red = pca.transform(x)
        return np.exp(x_red @ [bias])
    
    ## For blog and meps data use logarithm of top 2 PCs as tilting vars
    else: 
        np.random.seed(5)
        pca = decomposition.PCA(n_components=2)
        pca.fit(x_pca)
        x_red = pca.transform(x)
        x_red_min = abs(x_red.min(axis=0))
        x_red_adjusted = x_red + np.tile(x_red_min + 1, (len(x_red), 1))
        log_x_red_adjusted = np.log(x_red_adjusted)
        return np.exp(log_x_red_adjusted @ [-bias,bias])
    
    
def wsample(wts, n, d, frac=0.5):
    n = len(wts) ## n : length or num of weights
    indices = [] ## indices : vector containing indices of the sampled data
    normalized_wts = wts/max(wts)
    target_num_indices = int(n*frac)
    while(len(indices) < target_num_indices): ## Draw samples until have sampled ~25% of samples from D_test
        proposed_indices = np.where(np.random.uniform(size=n) <= normalized_wts)[0].tolist()
        ## If (set of proposed indices that may add is less than or equal to number still needed): then add all of them
        if (len(proposed_indices) <= target_num_indices - len(indices)):
            for j in proposed_indices:
                indices.append(j)
        else: ## Else: Only add the proposed indices that are needed to get to 25% of D_test
            for j in proposed_indices:
                if(len(indices) < target_num_indices):
                    indices.append(j)
    return(indices)

def exponential_tilting_indices(x_pca, x, dataset, bias=1):
    (n, d) = x.shape
    weights = get_w(x_pca, x, dataset, bias)
#     print("L1 squared : ", np.linalg.norm(weights, ord=1)**2)
#     print("L2 : ", np.linalg.norm(weights, ord=2)**2)
#     print("Effective sample size : ", np.linalg.norm(weights, ord=1)**2 / np.linalg.norm(weights, ord=2)**2)
    return wsample(weights, n, d)

def get_effective_sample_size(x_pca, x, dataset, bias=1):
    weights = get_w(x_pca, x, dataset, bias)
    sample_size_effective = int(np.linalg.norm(weights, ord=1)**2 / np.linalg.norm(weights, ord=2)**2)
    print("effective sample size : ", sample_size_effective)
    return sample_size_effective

######################################
# Define 3 regression algorithms
######################################

def leastsq_ridge(X,Y,X1,ridge_mult=0.001):
    lam = ridge_mult * np.linalg.svd(X,compute_uv=False).max()**2
    betahat = np.linalg.solve(\
            np.c_[np.ones(X.shape[0]),X].T.dot(np.c_[np.ones(X.shape[0]),X]) \
                              + lam * np.diag(np.r_[0,np.ones(X.shape[1])]),
            np.c_[np.ones(X.shape[0]),X].T.dot(Y))
    return betahat[0] + X1.dot(betahat[1:])

def random_forest(X,Y,X1,ntree=20):
    rf = RandomForestRegressor(n_estimators=ntree,criterion='mae').fit(X,Y)
    return rf.predict(X1)

def neural_net(X,Y,X1):
    nnet = MLPRegressor(solver='lbfgs',activation='logistic').fit(X,Y)
    return nnet.predict(X1)


def compute_PIs(X,Y,X1,alpha,fit_muh_fun, weights_full, dataset, bias):
    n = len(Y) ## Num training data
    n1 = X1.shape[0] ## Num test data (Note: This is larger than training to focus on confidence estimate)
#     n_effective = get_effective_sample_size(X, X, dataset, bias=1)
#     X_effective = X[0:n_effective, :]
#     Y_effective = Y[0:n_effective]

    
#     print("Y : ", Y[0:10])

    ###############################
    # Naive & jackknife/jack+/jackmm
    ###############################

    muh_vals = fit_muh_fun(X,Y,np.r_[X,X1])
    resids_naive = np.abs(Y-muh_vals[:n])
    muh_vals_testpoint = muh_vals[n:]
    resids_LOO = np.zeros(n)
    muh_LOO_vals_testpoint = np.zeros((n,n1))
    for i in range(n):
        muh_vals_LOO = fit_muh_fun(np.delete(X,i,0),np.delete(Y,i),\
                                   np.r_[X[i].reshape((1,-1)),X1])
        resids_LOO[i] = np.abs(Y[i] - muh_vals_LOO[0])
        muh_LOO_vals_testpoint[i] = muh_vals_LOO[1:]
    ind_q = (np.ceil((1-alpha)*(n+1))).astype(int)
    
    ###############################
    # Naive & jackknife/jack+/jackmm with effective sample size
    ###############################
#     resids_LOO_effective = resids_LOO[0:n_effective]
#     muh_LOO_vals_testpoint_effective = muh_LOO_vals_testpoint[0:n_effective]
#     ind_q_effective = (np.ceil((1-alpha)*(n_effective+1))).astype(int)

    
    
    ###############################
    # Weighted jackknife+
    ###############################
    ## DREW: Double check this later
    
    # Add infinity
    weights_normalized = np.zeros((n + 1, n1))
    sum_train_weights = np.sum(weights_full[0:n])
    for i in range(0, n + 1):
        for j in range(0, n1):
            if (i < n):
                weights_normalized[i, j] = weights_full[i] / (sum_train_weights + weights_full[n + j])
            else:
                weights_normalized[i, j] = weights_full[n+j] / (sum_train_weights + weights_full[n + j])


    unweighted_upper_vals = (muh_LOO_vals_testpoint.T + resids_LOO).T
    unweighted_lower_vals = (muh_LOO_vals_testpoint.T - resids_LOO).T
    
#     ## Add infty (distribution on augmented real line)
    positive_infinity = np.array([float('inf')])
    unweighted_upper_vals = np.vstack((unweighted_upper_vals, positive_infinity*np.ones(n1)))
    unweighted_lower_vals = np.vstack((unweighted_lower_vals, -positive_infinity*np.ones(n1)))
        
    y_upper_weighted = np.zeros(n1)
    y_lower_weighted = np.zeros(n1)
    
    for j in range(0, n1):
        y_upper_weighted[j] = weighted_quantile(unweighted_upper_vals[:, j], weights_normalized[:, j], 1 - alpha)
        y_lower_weighted[j] = weighted_quantile(unweighted_lower_vals[:, j], weights_normalized[:, j], alpha)
        
    
    ###############################
    # CV+
    ###############################

    K = 10
    n_K = np.floor(n/K).astype(int)
    base_inds_to_delete = np.arange(n_K).astype(int)
    resids_LKO = np.zeros(n)
    muh_LKO_vals_testpoint = np.zeros((n,n1))
    for i in range(K):
        inds_to_delete = (base_inds_to_delete + n_K*i).astype(int)
        muh_vals_LKO = fit_muh_fun(np.delete(X,inds_to_delete,0),np.delete(Y,inds_to_delete),\
                                   np.r_[X[inds_to_delete],X1])
        resids_LKO[inds_to_delete] = np.abs(Y[inds_to_delete] - muh_vals_LKO[:n_K])
        for inner_K in range(n_K):
            muh_LKO_vals_testpoint[inds_to_delete[inner_K]] = muh_vals_LKO[n_K:]
    ind_Kq = (np.ceil((1-alpha)*(n+1))).astype(int)



    ###############################
    # split conformal
    ###############################
    
    idx = np.random.permutation(n)
    n_half = int(np.floor(n/2))
    idx_train, idx_cal = idx[:n_half], idx[n_half:]
    muh_split_vals = fit_muh_fun(X[idx_train],Y[idx_train],np.r_[X[idx_cal],X1])
    resids_split = np.abs(Y[idx_cal]-muh_split_vals[:(n-n_half)])
    muh_split_vals_testpoint = muh_split_vals[(n-n_half):]
    ind_split = (np.ceil((1-alpha)*(n-n_half+1))).astype(int)


    ###############################
    # construct prediction intervals
    ###############################
        
    PIs_dict = {'naive' : pd.DataFrame(\
                    np.c_[muh_vals_testpoint - np.sort(resids_naive)[ind_q-1], \
                        muh_vals_testpoint + np.sort(resids_naive)[ind_q-1]],\
                           columns = ['lower','upper']),\
                'jackknife' : pd.DataFrame(\
                    np.c_[muh_vals_testpoint - np.sort(resids_LOO)[ind_q-1], \
                        muh_vals_testpoint + np.sort(resids_LOO)[ind_q-1]],\
                           columns = ['lower','upper']),\
                'jackknife+' : pd.DataFrame(\
                    np.c_[np.sort(muh_LOO_vals_testpoint.T - resids_LOO,axis=1).T[-ind_q], \
                        np.sort(muh_LOO_vals_testpoint.T + resids_LOO,axis=1).T[ind_q-1]],\
                           columns = ['lower','upper']),\
                'jackknife-mm' : pd.DataFrame(\
                    np.c_[muh_LOO_vals_testpoint.min(0) - np.sort(resids_LOO)[ind_q-1], \
                           muh_LOO_vals_testpoint.max(0) + np.sort(resids_LOO)[ind_q-1]],\
                           columns = ['lower','upper']),\
                'CV+' : pd.DataFrame(\
                    np.c_[np.sort(muh_LKO_vals_testpoint.T - resids_LKO,axis=1).T[-ind_Kq], \
                        np.sort(muh_LKO_vals_testpoint.T + resids_LKO,axis=1).T[ind_Kq-1]],\
                           columns = ['lower','upper']),\
                'split' : pd.DataFrame(\
                    np.c_[muh_split_vals_testpoint - np.sort(resids_split)[ind_split-1], \
                           muh_split_vals_testpoint + np.sort(resids_split)[ind_split-1]],\
                            columns = ['lower','upper']),\
               'JAW' : pd.DataFrame(\
                    np.c_[y_lower_weighted, \
                        y_upper_weighted],\
                           columns = ['lower','upper'])}
                
    return pd.concat(PIs_dict.values(), axis=1, keys=PIs_dict.keys())


## Function for obtaining predictive distribution (rather than interval)
def compute_PDs(X,Y,X1,alpha,fit_muh_fun, weights_full, dataset, bias):
    print("Computing predictive distributions")
    n = len(Y) ## Num training data
    n1 = X1.shape[0] ## Num test data (Note: This is larger than training to focus on confidence estimate)
#     n_effective = get_effective_sample_size(X, X, dataset, bias=1)
#     X_effective = X[0:n_effective, :]
#     Y_effective = Y[0:n_effective]

    
    ###############################
    # Naive & jackknife/jack+/jackmm
    ###############################

    muh_vals = fit_muh_fun(X,Y,np.r_[X,X1])
    resids_naive = np.abs(Y-muh_vals[:n])
    muh_vals_testpoint = muh_vals[n:]
    resids_LOO = np.zeros(n)
    muh_LOO_vals_testpoint = np.zeros((n,n1))
    for i in range(n):
        muh_vals_LOO = fit_muh_fun(np.delete(X,i,0),np.delete(Y,i),\
                                   np.r_[X[i].reshape((1,-1)),X1])
        resids_LOO[i] = np.abs(Y[i] - muh_vals_LOO[0])
        muh_LOO_vals_testpoint[i] = muh_vals_LOO[1:]
    ind_q = (np.ceil((1-alpha)*(n+1))).astype(int)
    
    ###############################
    # Naive & jackknife/jack+/jackmm with effective sample size
    ###############################
#     resids_LOO_effective = resids_LOO[0:n_effective]
#     muh_LOO_vals_testpoint_effective = muh_LOO_vals_testpoint[0:n_effective]
#     ind_q_effective = (np.ceil((1-alpha)*(n_effective+1))).astype(int)

    
    
    ###############################
    # Weighted jackknife+
    ###############################
    ## DREW: Double check this later
    
    # Add infinity
    weights_normalized = np.zeros((n + 1, n1))
    sum_train_weights = np.sum(weights_full[0:n])
    for i in range(0, n + 1):
        for j in range(0, n1):
            if (i < n):
                weights_normalized[i, j] = weights_full[i] / (sum_train_weights + weights_full[n + j])
            else:
                weights_normalized[i, j] = weights_full[n+j] / (sum_train_weights + weights_full[n + j])
                

    unweighted_upper_vals = (muh_LOO_vals_testpoint.T + resids_LOO).T
    unweighted_lower_vals = (muh_LOO_vals_testpoint.T - resids_LOO).T
    
#     ## Add infty
    positive_infinity = np.array([float('inf')])
    unweighted_upper_vals = np.vstack((unweighted_upper_vals, positive_infinity*np.ones(n1)))
    unweighted_lower_vals = np.vstack((unweighted_lower_vals, -positive_infinity*np.ones(n1)))
        
    y_upper_weighted = np.zeros(n1)
    y_lower_weighted = np.zeros(n1)
    
    for j in range(0, n1):
        y_upper_weighted[j] = weighted_quantile(unweighted_upper_vals[:, j], weights_normalized[:, j], 1 - alpha)
        y_lower_weighted[j] = weighted_quantile(unweighted_lower_vals[:, j], weights_normalized[:, j], alpha)
        
#     ###############################
#     # CV+
#     ###############################

    K = 10
    n_K = np.floor(n/K).astype(int)
    base_inds_to_delete = np.arange(n_K).astype(int)
    resids_LKO = np.zeros(n)
    muh_LKO_vals_testpoint = np.zeros((n,n1))
    for i in range(K):
        inds_to_delete = (base_inds_to_delete + n_K*i).astype(int)
        muh_vals_LKO = fit_muh_fun(np.delete(X,inds_to_delete,0),np.delete(Y,inds_to_delete),\
                                   np.r_[X[inds_to_delete],X1])
        resids_LKO[inds_to_delete] = np.abs(Y[inds_to_delete] - muh_vals_LKO[:n_K])
        for inner_K in range(n_K):
            muh_LKO_vals_testpoint[inds_to_delete[inner_K]] = muh_vals_LKO[n_K:]
    ind_Kq = (np.ceil((1-alpha)*(n+1))).astype(int)



#     ###############################
#     # split conformal
#     ###############################
    
    idx = np.random.permutation(n)
    n_half = int(np.floor(n/2))
    idx_train, idx_cal = idx[:n_half], idx[n_half:]
    muh_split_vals = fit_muh_fun(X[idx_train],Y[idx_train],np.r_[X[idx_cal],X1])
    resids_split = np.abs(Y[idx_cal]-muh_split_vals[:(n-n_half)])
    muh_split_vals_testpoint = muh_split_vals[(n-n_half):]
    ind_split = (np.ceil((1-alpha)*(n-n_half+1))).astype(int)

    

    ###############################
    # construct prediction intervals
    ###############################
    
    
    col_names = np.concatenate((['lower' + str(i) for i in range(0, n1)], ['upper' + str(i) for i in range(0, n1)]))
    
#     print(np.c_[np.sort(muh_LOO_vals_testpoint.T - resids_LOO,axis=1).T, \
#                          np.sort(muh_LOO_vals_testpoint.T + resids_LOO,axis=1).T])
    
    PDs_dict = {'naive' : pd.DataFrame(\
                    np.c_[np.sort(np.tile(muh_vals_testpoint, (n, 1)).T - np.tile(resids_naive, (n1, 1)),axis=1).T, \
                        np.sort(np.tile(muh_vals_testpoint, (n, 1)).T + np.tile(resids_naive, (n1, 1)),axis=1).T],\
                           columns = col_names),\
                'jackknife' : pd.DataFrame(\
                    np.c_[np.sort(np.tile(muh_vals_testpoint, (n, 1)).T - np.tile(resids_LOO, (n1, 1))).T, \
                        np.sort(np.tile(muh_vals_testpoint, (n, 1)).T + np.tile(resids_LOO, (n1, 1))).T],\
                           columns = col_names),\
                'jackknife+_sorted' : pd.DataFrame(\
                    np.c_[np.sort(muh_LOO_vals_testpoint.T - resids_LOO,axis=1).T, \
                        np.sort(muh_LOO_vals_testpoint.T + resids_LOO,axis=1).T],\
                           columns = col_names),\
                'jackknife+_not_sorted' : pd.DataFrame(\
                    np.c_[(muh_LOO_vals_testpoint.T - resids_LOO).T, \
                        (muh_LOO_vals_testpoint.T + resids_LOO).T],\
                           columns = col_names),\
                'CV+' : pd.DataFrame(\
                    np.c_[np.sort(muh_LKO_vals_testpoint.T - resids_LKO,axis=1).T, \
                        np.sort(muh_LKO_vals_testpoint.T + resids_LKO,axis=1).T],\
                           columns = col_names),\
                'split' : pd.DataFrame(\
                    np.c_[np.sort(np.tile(muh_split_vals_testpoint, (n_half, 1)).T - np.tile(resids_split, (n1, 1))).T, \
                           np.sort(np.tile(muh_split_vals_testpoint, (n_half, 1)).T + np.tile(resids_split, (n1, 1))).T],\
                            columns = col_names),\
               'weights_JAW_train' : pd.DataFrame(\
                    np.c_[weights_normalized[0:n, :], \
                        weights_normalized[0:n, :]],\
                           columns = col_names),\
               'weights_JAW_test' : pd.DataFrame(\
                    np.concatenate((weights_normalized[n, :], weights_normalized[n, :])).reshape((1, 2*n1)),\
                           columns = col_names),\
               'muh_vals_testpoint' : pd.DataFrame(\
                    np.concatenate((muh_vals_testpoint, muh_vals_testpoint)).reshape((1, 2*n1)),\
                           columns = col_names)}
    
                
    return PDs_dict
