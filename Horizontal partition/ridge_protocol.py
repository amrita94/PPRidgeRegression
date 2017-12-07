import time
import numpy as np
import pandas as pd
from math import log2, ceil

from MLE import MLEngine
from CSP import CSProvider
from DO import DOwner


def evaluate_secure_ridge_protocol(X, Y, X_test, Y_test, header, mag_l, num_DOs, lamda):
    """
    compute coefficients in secure_ridge_protocol.
    measure accuracy[error] and efficiency[runtime].
    
    :param X, Y, header: dataset info
    :param mag_l: the number of fractional digits. it is reason of truncate error
    :param num_DOs: the number of Data Owners
    :param lamda: the regulization parameter(lambda)
    :return : results of evaluation[errors, runtime for each step] and logs(N)
    """
    # assume that data is evenly distributed
    # split dataset as many as the number of Data Owners
    X_train = np.array_split(X, num_DOs)
    Y_train = np.array_split(Y, num_DOs)

    # compute optimal coefficients with a clean dataset
    w_clean = ridgeInTheClear(X, Y, lamda)

    '''Start Secure protocol computation'''
    # parameter setting
    numInstances = X.shape[0]
    numFeatures = X.shape[1]
    max_x = np.amax(X)
    max_y = np.amax(Y)
    magnitude = 10**mag_l # 10^l #(ex. l=3 -> magnitude = 1000) 

    # compute bound
    bound = compute_bound(numFeatures, numInstances, mag_l, max_x, max_y, lamda)
    n_length = max([bound, 2048])
    '''
    print("numInstances", numInstances, "numFeatures", numFeatures)
    print("max_x", max_x, "max_y",max_y)
    print("mag_l" ,mag_l, "lamda", lamda)
    print("n_length", n_length)
    '''

    # declare CSP & MLE & DOwners
    csp = CSProvider()
    mle = MLEngine()
    DOwners = []
    for index in range(num_DOs):
        DOwners.append(DOwner(X_train[index], Y_train[index], magnitude))

    print('Phase 1 : Protocol Horizontal')
    # declare Crypto Service Provider(CSP) & Phase1 step1
    start_merge1 = time.time()
    csp.key_gen(n_length)
    end_merge1 = time.time()

    # publish public_key to MLE & Downers
    mle.set_PK(csp.getPKey())
    for d_owner in DOwners:
        d_owner.set_PK(csp.getPKey())    

    # Phase1 step2
    start_merge2 = time.time()     
    for d_owner in DOwners:
        d_owner.computeEnc_Ab()
    end_merge2 = time.time()

    # Phase1 step3
    start_merge3 = time.time()
    for d_owner in DOwners:
        enc_A_i, enc_b_i = d_owner.getEnc_Ab()
        #mle merges enc_A_i, enc_b_i
        mle.add_enc_Ab(enc_A_i, enc_b_i, lamda, magnitude)
    end_merge3 = time.time()

    print('Phase 2 : Protocol Ridge')
    # Phase2 step1
    enc_C, enc_d = mle.protocol_ridge_step1()
    after_step1 = time.time()        

    # Phase2 step2
    w_tilda = csp.protocol_ridge_step2(enc_C, enc_d)
    after_step2 = time.time()

    # Phase2 step3
    w_star = mle.protocol_ridge_step3(w_tilda)
    after_step3 = time.time()

    ''' Evaluate performance and accuracy '''
    # compute 2-norm
    assert w_clean.shape == w_star.shape
    Norm = (np.linalg.norm(np.add(w_clean, -w_star)))

    #Error_train = evaluate_accuracy(X, Y, lamda, w_clean, w_star)
    #Error_test = evaluate_accuracy(X_test, Y_test, lamda, w_clean, w_star)
    MSE_clean, MSE_star, MSE_Error = MSE(X_test, Y_test, w_clean, w_star)
    MAE_clean, MAE_star, MAE_Error = MAE(X_test, Y_test, w_clean, w_star)

    # compute Runtime
    Runtime_Merge_step1= (end_merge1 - start_merge1)
    Runtime_Merge_step2= ((end_merge2 - start_merge2)/(num_DOs))
    Runtime_Merge_step3= (end_merge3 - start_merge3)  
    Runtime_ridge_step1= (after_step1 - end_merge3)
    Runtime_ridge_step2= (after_step2 - after_step1)
    Runtime_ridge_step3= (after_step3 - after_step2)    

    # print results on the console
    print('Evaluate performance and accuracy')
    print(pd.DataFrame({'feature': header[:len(w_clean)], 'w(by LR-alg)': w_clean, 'w(by MLE(PPLR))': w_star}))
    print('2norm:', Norm)
    #print('Error(trainset):', Error_train)
    #print('Error(testset):', Error_test)
    print("MSE(clean, star, normalize) = ", MSE_clean, MSE_star, MSE_Error)
    print("MAE(clean, star, normalize) = ", MAE_clean, MAE_star, MAE_Error)
    print('Runtime(Phase1-step1) = ', Runtime_Merge_step1)
    print('Runtime(Phase1-step2) = ', Runtime_Merge_step2)
    print('Runtime(Phase1-step3) = ', Runtime_Merge_step3)
    print('Runtime(Phase2-step1) = ', Runtime_ridge_step1)
    print('Runtime(Phase2-step2) = ', Runtime_ridge_step2)
    print('Runtime(Phase2-step3) = ', Runtime_ridge_step3)

    return Norm, MSE_clean, MSE_star, MSE_Error, MAE_clean, MAE_star, MAE_Error, Runtime_Merge_step1,Runtime_Merge_step2,Runtime_Merge_step3, Runtime_ridge_step1,Runtime_ridge_step2,Runtime_ridge_step3, n_length


def MSE(X, Y, w_clean, w_star):
    num_instance = X.shape[0]
    try:
        # Error = | (F(w_tilda) - F(w)) / F(w) |
        # MSE(w) = norm(yi-xi*w)^2 / n
        MSE_clean = (np.linalg.norm(Y - np.matmul(X, w_clean))**2) /num_instance
        MSE_star = (np.linalg.norm(Y - np.matmul(X, w_star))**2) /num_instance
        MSE_Error = np.absolute(MSE_star - MSE_clean) / MSE_clean
    except:
        print('w_clean ', w_clean)
        print('w_star ' , w_star )
        assert False
    return MSE_clean, MSE_star, MSE_Error


def MAE(X, Y, w_clean, w_star):
    num_instance = X.shape[0]
    try:
        # Error = | (F(w_tilda) - F(w)) / F(w) |
        # MSE(w) = norm(yi-xi*w)^2 / n
        MAE_clean = np.linalg.norm(Y - np.matmul(X, w_clean), 1) /num_instance
        MAE_star = np.linalg.norm(Y - np.matmul(X, w_star), 1) /num_instance
        MAE_Error = ( np.absolute(MAE_star - MAE_clean) / MAE_clean )
    except:
        print('w_clean ', w_clean)
        print('w_star ' , w_star )
        assert False
    return MAE_clean, MAE_star, MAE_Error


def evaluate_accuracy(X, Y, lamda, w_clean, w_star):
    num_instance = X.shape[0]
    try:
        # Error = | (F(w_tilda) - F(w)) / F(w) |
        # F(w) = norm(yi-xi*w)^2 + lamda*norm(w)^2
        F_w_clean = (np.linalg.norm(Y - np.matmul(X, w_clean))**2) + lamda*(np.linalg.norm(w_clean)**2)
        F_w_star = (np.linalg.norm(Y - np.matmul(X, w_star))**2) + lamda*(np.linalg.norm(w_star)**2)
        Error = ( np.absolute(F_w_star - F_w_clean) / F_w_clean )
        #print("F_w_clean", F_w_clean, "F_w_star", F_w_star, "Error", Error)
    except:
        print('w_clean ', w_clean)
        print('w_star ' , w_star )
        assert False
    return Error


def ridgeInTheClear(X, Y, lamda):
    """
    compute coefficients in the clear(without secure protocol).
    use the closed-form of ridge regression.
    
    :param X, Y: training dataset
    :return w_star: computed coefficients = inv(trans(X)*X + lamda*I) * trans(X) * Y
    """
    #step 1
    A = np.matmul(np.transpose(X), X)
    b = np.matmul(np.transpose(X), Y)

    #step 2
    num_dimensions = A.shape[0]
    M = A + ( lamda * np.eye(num_dimensions) )

    #step 3
    w_star = np.matmul(np.linalg.inv(M), b)

    return w_star


def compute_bound(d, n, l, max_x, max_y, lamda):
    """
    compute bound to decide the size of plaintext space(log2(N))
    
    :param d: the number of dimensions of dataset
    :param n: the number of instances of dataset
    :param l: the fractional digit
    :param max_x, max_y: maximum values in dataset(X, Y)
    :return bound: log2(N)
    """
    bound = 1 + log2(d) + (d-1)/2*log2(d-1)+ 4*l*d*log2(10) + (2*d-1)*log2(n*(max_x**2) + lamda)+ log2(n) + log2(max_x) + log2(max_y)
    bound = ceil(bound)

    # the bound should be an even number to reduce the key generation time. (N=pq)
    if bound % 2 == 0:
        return bound
    else: 
        return bound+1



