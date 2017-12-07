import numpy as np
import argparse
import copy
import csv
import os
from datetime import date
from math import ceil

from dataprocessing import genDataset
from ridge_protocol import evaluate_secure_ridge_protocol


# read parameter values from a command line 
parser = argparse.ArgumentParser(description='Todo', epilog='Usage : python3 exp_syn.py ')
args = parser.parse_args()  


if __name__ == '__main__':

    # create csv file to save experiment results
    save_path = os.path.join(os.getcwd(), 'experiment_result')
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok = False)

    result_filename = os.path.join(save_path, str(date.today())+'(hor,syn).csv')
    print("Generating ", result_filename)
    with open(result_filename, 'w') as csvfile:
        writer = csv.writer(csvfile, dialect = 'excel')
        writer.writerow(['Num_instances', 'Num_dimensions', 'Magnitude', 'log2(N)',  '2norm', 'MSE(clean)', 'MSE(secure)', 'MSE(Error)', 'MAE(clean)', 'MAE(secure)', 'MAE(Error)', 'Runtime(Phase1-step1)', 'Runtime(Phase1-step2)', 'Runtime(Phase1-step3)','Runtime(Phase2-step1)','Runtime(Phase2-step2)','Runtime(Phase2-step3)'])

    #parameters setting
    lamda = 0.1
    num_DOs = 10
    for num_instances in [10**3, 10**4, 10**5]:
        for num_dimensions in [10,20,30,40]:
            for mag_l in [3, 4, 5]:

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

                for iter_index in range(5): # set iteration
                    print("\n\niteration:", iter_index)

                    # generate a dataset
                    X, Y, header = genDataset(num_instances, num_dimensions)
                    X_test, Y_test, _ = genDataset(num_instances//10, num_dimensions)

                    # do the experiment
                    norm, mse_clean, mse_star, mse_error, mae_clean, mae_star, mae_error, run_11, run_12, run_13, run_21, run_22, run_23, n_length = evaluate_secure_ridge_protocol(X, Y, X_test, Y_test, header, mag_l, num_DOs, lamda)

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

