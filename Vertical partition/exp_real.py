import numpy as np
import argparse
import copy
import csv
import os
from datetime import date
from math import ceil

from ridge_protocol import evaluate_secure_ridge_protocol
from dataprocessing import load_data


# read parameter values from a command line 
#parser = argparse.ArgumentParser(description='Todo', epilog='Usage Example : python3 exp_real.py -d {forest|energy|boston|blog|air|facebook|beijing|wine|student|bike}')
#parser.add_argument('-d', '--dataset', type=str, help='The name of dataset')
parser = argparse.ArgumentParser(description='Todo', epilog='Usage Example : python3 exp_real.py')
args = parser.parse_args()  


if __name__ == '__main__':

    # create csv file to save experiment results
    save_path = os.path.join(os.getcwd(), 'experiment_result')
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok = False)

    result_filename = os.path.join(save_path, str(date.today())+'(Ver,real).csv')
    print("Generating ", result_filename)
    with open(result_filename, 'w') as csvfile:
        writer = csv.writer(csvfile, dialect = 'excel')
        writer.writerow(['Num_instances', 'Num_dimensions', 'Magnitude', 'log2(N)',  '2norm', 'MSE(clean)', 'MSE(secure)', 'MSE(Error)', 'MAE(clean)', 'MAE(secure)', 'MAE(Error)', 'Runtime(Phase1-step1)', 'Runtime(Phase1-step2)', 'Runtime(Phase1-step3)','Runtime(Phase2-step1)','Runtime(Phase2-step2)','Runtime(Phase2-step3)'])

    #parameters setting
    lamda = 0.1
    num_DOs = 10
    for name_data in [ "forest","energy","boston","blog","air","facebook","beijing","wine","student","bike" ]:
        # load dataset           
        X, Y, header = load_data(name_data)
        num_instances =  X.shape[0]
        num_dimensions = X.shape[1]

        for mag_l in [1,2,3,4]:

            print('\nnum_dimensions:', num_dimensions)
            print('num_instances:', num_instances )
            print('mag_l:', mag_l, '\n')
            print('num_DOs:', num_DOs)

            # for protocol analysis
            storage_2norm = []
            storage_MSE_clean = []
            storage_MSE_secure = []
            storage_MSE_error = []
            storage_MAE_clean = []
            storage_MAE_secure = []
            storage_MAE_error = []
            storage_log_N = []
            storage_Runtime_Merge_step1 = []
            storage_Runtime_Merge_step2 = []
            storage_Runtime_Merge_step3 = []
            storage_Runtime_ridge_step1 = []
            storage_Runtime_ridge_step2 = []
            storage_Runtime_ridge_step3 = []

            for iter_index in range(1): # set iteration
                # do the experiment
                norm, mse_clean, mse_star, mse_error, mae_clean, mae_star, mae_error, run_11, run_12, run_13, run_21, run_22, run_23, n_length = evaluate_secure_ridge_protocol(X, Y, X, Y, header, mag_l, num_DOs, lamda)

                # save a result
                storage_2norm.append(norm)
                storage_MSE_clean.append(mse_clean)
                storage_MSE_secure.append(mse_star)
                storage_MSE_error.append(mse_error)
                storage_MAE_clean.append(mae_clean)
                storage_MAE_secure.append(mae_star)
                storage_MAE_error.append(mae_error)
                storage_log_N.append(n_length)
                storage_Runtime_Merge_step1.append(run_11)
                storage_Runtime_Merge_step2.append(run_12)
                storage_Runtime_Merge_step3.append(run_13)
                storage_Runtime_ridge_step1.append(run_21)
                storage_Runtime_ridge_step2.append(run_22)
                storage_Runtime_ridge_step3.append(run_23)

            # save experiment results into csv file
            with open(result_filename, 'a', newline = '') as csvfile:
                writer = csv.writer(csvfile, dialect = 'excel')
                writer.writerow([num_instances, num_dimensions, mag_l, ceil(np.average(storage_log_N)), np.average(storage_2norm), np.average(storage_MSE_clean), np.average(storage_MSE_secure), np.average(storage_MSE_error), np.average(storage_MAE_clean), np.average(storage_MAE_secure), np.average(storage_MAE_error), np.average(storage_Runtime_Merge_step1), np.average(storage_Runtime_Merge_step2), np.average(storage_Runtime_Merge_step3), np.average(storage_Runtime_ridge_step1), np.average(storage_Runtime_ridge_step2), np.average(storage_Runtime_ridge_step3)])

            if np.average(storage_MSE_error) < 10**(-6):
                break




