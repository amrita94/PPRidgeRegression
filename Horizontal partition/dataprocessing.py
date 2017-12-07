import numpy as np
import csv
#import pandas as pd
from copy import copy


# define paths where save data files.
forestFire_file_path = '../dataset/forestfires.csv'
energy_file_path = '../dataset/energydata_complete.csv'
blog_file_path = '../dataset/blogData.csv'
air_file_path = '../dataset/AirQualityUCI.csv'
facebook_file_path = '../dataset/facebook.csv'
beijing_file_path = '../dataset/beijing.csv'
crime_file_path = '../dataset/crime.data'
wine_file_path = '../dataset/winequality-white.csv'
student_file_path = '../dataset/student-perform.csv'
bike_file_path = '../dataset/bike.csv'
syn5_file_path = '../dataset/syn/syndataset5.csv'
syn10_file_path = '../dataset/syn/syndataset10.csv'
syn15_file_path = '../dataset/syn/syndataset15.csv'


# generate random dataset
def genDataset(num_instances, num_features):
    # let n = num_instances, d = num_features
    # generate uniformly at random X [-1.0, 1.0] #X = (n,d) matrix
    X = (np.random.rand(num_instances, num_features) -0.5) *2
    #X = (np.random.rand(num_instances, num_features) ) #[0, 1]

    # generate uniformly at random beta [-1.0, 1.0] # beta = (d)vector
    beta = (np.random.rand(num_features) -0.5) *2

    # generate gaussian noise vector N(0, 1)
    e = np.random.normal(size = num_instances)

    # compute Y = X*beta + e
    Y = np.add(np.matmul(X, beta), e)
    # define bound for Y
    bound = 2
    Y[Y > bound] = bound   #maximum bound
    Y[Y < (-bound)] = -bound #minimum bound

    # generate feature name
    header = []
    for index in range(num_features):
        header.append('feature'+str(index))

    return X, Y, header


# load a selected dataset 
def load_data(selected_data):
    if selected_data == 'forest':
        return load_forestFire(forestFire_file_path)
    elif selected_data == 'energy':
        return load_energydata(energy_file_path)
    elif selected_data == 'boston':
        return load_bostondata()   
    elif selected_data == 'blog':
        return load_blogdata(blog_file_path)
    elif selected_data == 'air':
        return load_airdata(air_file_path)
    elif selected_data == 'facebook':
        return load_facebookdata(facebook_file_path)
    elif selected_data == 'beijing':
        return load_beijingdata(beijing_file_path)
    elif selected_data == 'crime':
        return load_crimedata(crime_file_path)
    elif selected_data == 'wine':
        return load_winedata(wine_file_path)
    elif selected_data == 'student':
        return load_studentdata(student_file_path)
    elif selected_data == 'bike':
        return load_bikedata(bike_file_path)
    # use these synthetic dataset[syn5, syn10, syn15] for testing Yao Implementation
    elif selected_data == 'syn5':
        return load_syndata(syn5_file_path)
    elif selected_data == 'syn10':
        return load_syndata(syn10_file_path)
    elif selected_data == 'syn15':
        return load_syndata(syn15_file_path)
    elif selected_data == 'medical':
        return load_medicaldata(medical_file_path)
    else:
        print('Error during load_data')
        return None



# define functions to load datasets
# Each real dataset needs to be preprocessed differently
# [categorial-numeric conversion, drop instances which have missing values, different structures of csv file].

def load_blogdata(file_name):
    #read data from CSV file
    data = None
    with open(file_name, 'rt') as datas_csv:
        reader = csv.reader(datas_csv, delimiter=',')
        data = np.array(list(reader))

    data = np.transpose(data)
    X = np.transpose(data[:-1])
    Y = np.transpose(data[-1])
    
    #convert data_type(string -> float)
    X = X.astype(float)
    Y = Y.astype(float)
    
    #generate feature name
    header = []
    for index in range(X.shape[1]):
        header.append('feature'+str(index))

    print('dataset = blog dataset\n')
    
    return X, Y, header


def load_bostondata():
    from sklearn.datasets import load_boston

    dataset = load_boston()
    header = dataset.feature_names
    #datas = pd.DataFrame(dataset.data)
    datas.columns = header
    X = datas
    Y = dataset.target

    #convert data_type(string -> float)
    X = np.asarray(X)
    Y = np.asarray(Y)
    X = X.astype(float)
    Y = Y.astype(float)

    print('dataset = bostondata\n')    
    return X, Y, header


def load_energydata(file_name):
    #read data from CSV file
    data = None
    with open(file_name, 'rt') as datas_csv:
        reader = csv.reader(datas_csv, delimiter=',')
        data = np.array(list(reader))
    header = data[0]
    # keep feature name
    header = header[1:]

    data = np.transpose(data[1:])
    X = np.transpose(data[1:])
    Y = np.transpose(data[0])
    
    #convert data_type(string -> float)
    X = X.astype(float)
    Y = Y.astype(float)

    print('dataset = energy dataset\n')
    return X, Y, header


def load_airdata(file_name):
    #read data from CSV file
    data = None
    with open(file_name, 'rt') as datas_csv:
        reader = csv.reader(datas_csv, delimiter=',')
        data = np.array(list(reader))
    header = data[0][:-1]
    data = np.transpose(data[1:])
    X = np.transpose(data[:-1])
    Y = np.transpose(data[-1])
    
    #convert data_type(string -> float)
    X = X.astype(float)
    Y = Y.astype(float)
    
    #generate feature name
    print('dataset = blog dataset\n')
    
    return X, Y, header


def load_facebookdata(file_name):
    #read data from CSV file
    data = None
    with open(file_name, 'rt') as datas_csv:
        reader = csv.reader(datas_csv, delimiter=',')
        data = np.array(list(reader))
    header = data[0][:-1]
    data = np.transpose(data[1:])
    X = np.transpose(data[:-1])
    Y = np.transpose(data[-1])
    
    #convert data_type(string -> float)
    X = X.astype(float)
    Y = Y.astype(float)
    
    #generate feature name
    print('dataset = blog dataset\n')
    
    return X, Y, header


def load_forestFire(file_name):
    #read data from CSV file
    data = None
    with open(file_name, 'rt') as datas_csv:
        reader = csv.reader(datas_csv, delimiter=',')
        data = np.array(list(reader))
    header = data[0]
    data = np.transpose(data[1:])
    X = np.transpose(data[:-1])
    Y = np.transpose(data[-1])
    
    # convert categorical data to numerical representation(month, week)
    month_dic = {'jan':1, 'feb':2, 'mar':3,
                 'apr':4, 'may':5, 'jun':6,
                 'jul':7, 'aug':8, 'sep':9,
                 'oct':10, 'nov':11, 'dec':12
                 }
    week_dic = {'mon':1, 'tue':2, 'wed':3, 
                'thu':4, 'fri':5, 'sat':6, 'sun':7
                }
    # to convert categorical data to numerical representation
    def cateToNumeric(dictionary, data, f_index):
        num_instances = data.shape[0]
        for i_index in range(num_instances):
            data[i_index][f_index] = dictionary[data[i_index][f_index]]

    cateToNumeric(month_dic, X, 2)
    cateToNumeric(week_dic, X, 3)   
    
    #convert data_type(string -> float)
    X = X.astype(float)
    Y = Y.astype(float)

    print('dataset = forestfire\n')
    return X, Y, header


def load_beijingdata(file_name):
    #read data from CSV file
    data = None
    with open(file_name, 'rt') as datas_csv:
        reader = csv.reader(datas_csv, delimiter=',')
        data = np.array(list(reader))
    header = data[0][:-1]
    data = np.transpose(data[1:])
    X = np.transpose(data[:-1])
    Y = np.transpose(data[-1])
    
    # if actual value is 'NA', remove it from dataset
    NA_list = []
    for index, item in enumerate(Y):
        if item == 'NA':
            NA_list.append(index)
            
    X = np.delete(X, np.asarray(NA_list), axis=0)
    Y = np.delete(Y, np.asarray(NA_list))
    
    X = np.transpose(X)
    temp_X = X[-1]
    temp_X = categorical_to_numerical(temp_X)
    temp_X = one_hotencode(temp_X)
    temp_X = np.transpose(temp_X)
    X = np.concatenate((X[:-1], temp_X), axis=0)

    X = np.transpose(X)
    feature_name = header[-1]
    header = header[:-1]
    for index in range(len(X[0]) - len(header)):
            header = np.append(header, [feature_name+str(index)])

    #convert data_type(string -> float)
    X = X.astype(float)
    Y = Y.astype(float)
    
    #generate feature name
    print('dataset = beijing dataset\n')
    
    return X, Y, header


def load_crimedata(file_name):
    #read data from CSV file
    data = None
    with open(file_name, 'rt') as datas_csv:
        reader = csv.reader(datas_csv, delimiter=',')
        data = np.array(list(reader))

    data = np.transpose(data[1:])
    X = np.transpose(data[:-1])
    Y = np.transpose(data[-1])
    X[X == '?'] = 0 
    
    X = np.transpose(X)
    
    temp_X = np.asarray(categorical_to_numerical(X[3]))
    temp_X = np.reshape(temp_X, (1, temp_X.shape[0]))
    X = np.delete(X, 3, axis=0)
    X = np.concatenate((X, temp_X), axis=0)
    X = np.transpose(X)
        
    #generate feature name
    header = []
    for index in range(X.shape[1]):
        header.append('feature'+str(index))
    #convert data_type(string -> float)
    X = X.astype(float)
    Y = Y.astype(float)
    
    #generate feature name
    print('dataset = crime dataset\n')
    
    return X, Y, header


def load_winedata(file_name):
    #read data from CSV file
    data = None
    with open(file_name, 'rt') as datas_csv:
        reader = csv.reader(datas_csv, delimiter=';')
        data = np.array(list(reader))
    header = data[0][:-1]
    data = np.transpose(data[1:])
    X = np.transpose(data[:-1])
    Y = np.transpose(data[-1])
    
    #convert data_type(string -> float)
    X = X.astype(float)
    Y = Y.astype(float)
    
    #generate feature name
    print('dataset = wine dataset\n')
    
    return X, Y, header

def load_studentdata(file_name):
    #read data from CSV file
    data = None
    with open(file_name, 'rt') as datas_csv:
        reader = csv.reader(datas_csv, delimiter=',')
        data = np.array(list(reader))
    header = data[0][:-1]
    data = np.transpose(data[1:])
    X = np.transpose(data[:-1])
    Y = np.transpose(data[-1])
    
    
    X = np.transpose(X)
    for index in [0,1,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22]:
        X[index] = categorical_to_numerical(X[index])
    X = np.transpose(X)
    
    #convert data_type(string -> float)
    X = X.astype(float)
    Y = Y.astype(float)
    
    #generate feature name
    print('dataset = student dataset\n')
    
    return X, Y, header


def load_bikedata(file_name):
    #read data from CSV file
    data = None
    with open(file_name, 'rt') as datas_csv:
        reader = csv.reader(datas_csv, delimiter=',')
        data = np.array(list(reader))
    header = data[0][:-1]
    data = np.transpose(data[1:])
    X = np.transpose(data[:-1])
    Y = np.transpose(data[-1])
    
    X = np.transpose(X)
    X = np.delete(X,[0,1], axis=0)
    X = np.transpose(X)
    
    #convert data_type(string -> float)
    X = X.astype(float)
    Y = Y.astype(float)
    
    #generate feature name
    print('dataset = bike dataset\n')
    
    return X, Y, header


def load_syndata(file_name):
    #read data from CSV file
    data = None
    with open(file_name, 'rt') as datas_csv:
        reader = csv.reader(datas_csv, delimiter=',')
        data = np.array(list(reader))

    data = np.transpose(data)
    X = np.transpose(data[:-1])
    Y = np.transpose(data[-1])
    
    #convert data_type(string -> float)
    X = X.astype(float)
    Y = Y.astype(float)
    
    #generate feature name
    header = []
    for index in range(X.shape[1]):
        header.append('feature'+str(index))

    print('dataset =' + file_name + '\n')

    return X, Y, header


# encode categorical features to numerical values
def categorical_to_numerical(X):
    item_list = list(set(X))
    for index, item in enumerate(item_list):
        X[X==item] = index
    return X
        

# encode categorical integer features using one-hot aka one-K scheme.
def one_hotencode(X, num_classes=None):
    X = np.array(X, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(X) + 1
    n = X.shape[0]
    categorical = np.zeros((n, num_classes), dtype=int)
    categorical[np.arange(n), X] = 1
    return categorical


# medical_train_path = '../dataset/medical/medical_train.csv'
# medical_test_path = '../dataset/medical/medical_test.csv'
def load_medicaldata(train_file, test_file):
    data = None
    #load traindata from CSV file
    with open(train_file, 'rt') as datas_csv:
        reader = csv.reader(datas_csv, delimiter=',')
        data = np.array(list(reader))
    header = data[0]

    data = np.transpose(data[1:])
    X_train = np.transpose(data[:-1])
    Y_train = np.transpose(data[-1])
    
    #convert data_type(string -> float)
    X_train = X_train.astype(float)
    Y_train = Y_train.astype(float)
    
    #load traindata from CSV file
    with open(test_file, 'rt') as datas_csv:
        reader = csv.reader(datas_csv, delimiter=',')
        data = np.array(list(reader))

    data = np.transpose(data[1:])
    X_test = np.transpose(data[:-1])
    Y_test = np.transpose(data[-1])
    
    #convert data_type(string -> float)
    X_test = X_test.astype(float)
    Y_test = Y_test.astype(float)

    print('dataset = preprocessed medical dataset' + '\n')

    return X_train, Y_train, X_test, Y_test, header


