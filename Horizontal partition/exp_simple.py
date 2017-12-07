import numpy as np
import argparse

from dataprocessing import genDataset
from ridge_protocol import evaluate_secure_ridge_protocol


# read parameter values from a command line 
parser = argparse.ArgumentParser(description='Todo', epilog='Usage Example : python3 exp_simple.py -ni 1100 -nd 5')
parser.add_argument('-ni', '--num_instances', type=int, help='the number of instances in a dataset')
parser.add_argument('-nd', '--num_dimensions', type=int, help='the number of dimensions(features) in a dataset')
args = parser.parse_args()


if __name__ == '__main__':

    #parameters setting 
    lamda = 0.1
    num_DOs = 10
    mag_l = 3

    print('\nnum_dimensions:', args.num_dimensions)
    print('num_instances:', args.num_instances )
    print('mag_l:', mag_l, '\n')
    print('num_DOs:', num_DOs)

    # load dataset
    X, Y, header = genDataset(args.num_instances, args.num_dimensions)
    X_test, Y_test, _ = genDataset(args.num_instances, args.num_dimensions)
    
    # evaluate the protocol
    norm, mse_clean, mse_star, mse_error, mae_clean, mae_star, mae_error, run_11, run_12, run_13, run_21, run_22, run_23, n_length = evaluate_secure_ridge_protocol(X, Y, X_test, Y_test, header, mag_l, num_DOs, lamda)


