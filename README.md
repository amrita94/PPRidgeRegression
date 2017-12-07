# Privacy Preserving Ridge Regression

This repository contains the python-based implementation for the [paper](https://eprint.iacr.org/2017/979). 

We notify that this project has been under continual development.


# Setting up

### Dependencies

This uses [gmpy2](https://pypi.python.org/pypi/gmpy2) for efficient arithmetic operations with large integers. gmpy2 requires GMP 5.0.x or later, MPFR 3.1.x or later, MPC 1.0.1 or later. The detail instruction for installing gmpy2 is in the [link](https://gmpy2.readthedocs.io/en/latest/intro.html#installing-gmpy2-on-unix-linux)

Also, this requires [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html), [sklearn](http://scikit-learn.org/stable/), [scipy](https://www.scipy.org/) for data processing. Installing these libraries is therefore a pre-requisite. These libraries will take care of other dependences like [numpy](http://www.numpy.org/), and so on.

```
pip install gmpy2
pip install pandas sklearn scipy
```


### Installation

Once dependencies have been taken care of, you can install the codes using
`pip` or by cloning this Github repository.

#### `pip` installation

```
pip install -e git+https://github.com/ykw6644/PPRidgeRegression.git
```

#### cloning repository

```
git clone https://github.com/ykw6644/PPRidgeRegression.git
```

### Currently supported setups

Although the codes are likely to work on many other machine configurations, we notify that we
currently test it using Python3.5.2 on Ubuntu 16.04 LTS and Scientific Linux7.

# Running sample experiments

To help you get started with our protocol, 
we support the sample experiments using different types of datasets and parameter setting.
We fix some of parameters to simplify experiments. We assume that 10 DataOwners are in Horizontal cases, 5 DataOwners are in Vertical cases. Additionally, we fix the regularization parameter to 0.1. The detail information and our experimental results are described in the [paper](https://eprint.iacr.org/2017/979). 

We assume that you run codes at project home directory.

* **Synthetic Dataset/Horizontal/singe parameter setting** ([code](https://github.com/ykw6644/PPRidgeRegression/blob/master/Horizontal%20partition/exp_simple.py)): 
This runs our protocol on a horizontally partitioned synthetic dataset. It requires two parameters(the total number of data point, the number of features in each data point) as inputs. Run ``` python3 Horizontal/exp_simple.py -ni 1000 -nd 20 ```. In the example, we use totally 1000 datapoints which have 20 features. Thus, each dataowner has 100 datapoints. 

* **Synthetic Dataset/Horizontal/multi parameter setting** ([code](https://github.com/ykw6644/PPRidgeRegression/blob/master/Horizontal%20partition/exp_syn.py)): 
To evaluate the effect of the parameters on our system’s performance, we run experiments on horizontally partitioned synthetic datasets with multiple paremeter setting. Run ``` python3 Horizontal/exp_syn.py ```. An experimental result\[date-(hor,syn).csv\] will be stored in Horizontal/experimental_result directory. To accurately measure, values in the result are averaged on 5 repetition. In this experiment, we use \[1000, 10000, 100000\], \[10,20,30,40\], \[3,4,5\] as the number of datapoints, the number of features in each datapoint, the fractional digits respectively.

* **UCI Dataset/Horizontal** ([code](https://github.com/ykw6644/PPRidgeRegression/blob/master/Horizontal%20partition/exp_real.py)): This runs our protocol on 10 real-world datasets downloaded from the UCI repository([link](https://archive.ics.uci.edu/ml/datasets.html)). For each dataset, we removed the data points with missing values and we use 1-of-k encoding to convert nominal features to numerical ones. In addition, We choose small integers as fractional digits to have a negligible error rate without degrading our system efficiency. Run ``` python3 Horizontal/exp_real.py ```. The experimental result\[date-(hor,real).csv\] will be stored in Horizontal/experimental_result directory. 

* **Synthetic Dataset/Vertical/singe parameter setting** ([code](https://github.com/ykw6644/PPRidgeRegression/blob/master/Vertical%20partition/exp_simple.py)): 
This runs our protocol on a vertically partitioned synthetic dataset. It requires two parameters(the total number of data point, the number of features in each data point) as inputs. Run ``` python3 Vertical/exp_simple.py -ni 1000 -nd 20 ```. In the example, we use totally 1000 datapoints which have 20 features. Thus, each dataowner has 200 datapoints. 

* **Synthetic Dataset/Vertical/multi parameter setting** ([code](https://github.com/ykw6644/PPRidgeRegression/blob/master/Vertical%20partition/exp_syn.py)): To evaluate the effect of the parameters on our system’s performance, we run experiments on vertically partitioned synthetic datasets with multiple paremeter setting. Run ``` python3 Vertical/exp_syn.py ```. An experimental result\[date-(ver,syn).csv\] will be stored in Vertical/experimental_result directory. To accurately measure, values in the result are averaged on 5 repetition. In this experiment, we use \[1000, 2000, 3000\], \[10,15,20\], \[3,4,5\] as the number of datapoints, the number of features in each datapoint, the fractional digits respectively. 

* **UCI Dataset/Vertical** ([code](https://github.com/ykw6644/PPRidgeRegression/blob/master/Vertical%20partition/exp_real.py)): This runs our protocol on 10 real-world datasets downloaded from the UCI repository([link](https://archive.ics.uci.edu/ml/datasets.html)). For each dataset, we removed the data points with missing values and we use 1-of-k encoding to convert nominal features to numerical ones. In addition, We choose small integers as fractional digits to have a negligible error rate without degrading our system efficiency. Run ``` python3 Vertical/exp_real.py ```. The experimental result\[date-(ver,real).csv\] will be stored in Vertical/experimental_result directory. 


# Additional Info

## Contributing

Contributions are welcomed! 


## Citing this work

If you use this implementation for academic research, you are highly encouraged to cite the following [paper](https://eprint.iacr.org/2017/979):


## Contributors(Authors)

* Irene Giacomelli(University of Wisconsin-Madison, Madison)
* Somesh Jha(University of Wisconsin-Madison, Madison)
* Marc Joye(NXP Semiconductors, San Jose)
* C. David Page(University of Wisconsin-Madison, Madison)
* Kyonghwan Yoon(University of Wisconsin-Madison, Madison)

## Copyright

Copyright 2017 - University of Wisconsin-Madison
