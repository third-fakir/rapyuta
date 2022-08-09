# import library you made (regression) and 3rd party library here
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from regression.plot import plot_regression_results
from regression.testing import testing
from regression.training import training
from regression.load_csv import load_csv

if __name__ == "__main__":
    # implement your application code here
    idx = "ID"
    filename = "data/regression_train.csv"
    train_data = load_csv(filename,idx)
    
    scatter_matrix(train_data, alpha=0.5, figsize=(10, 10), diagonal='hist');
    train_data_n = (train_data - train_data.mean()) /train_data.std()
    train_data_n.describe().transpose()
    train_data_n.plot.box()

    pred, err, model = training(train_data)

    fig, ax = plt.subplots(figsize=(8, 8))  
    plot_regression_results(ax, train_data.loc[:,["BicepC"]].values, pred,'LinearRegression','MSE={:.2f} cm'.format(err),"BicepC")
    plt.tight_layout()
    plt.show()

    testfile = "data/regression_test.csv"
    test_data = load_csv(testfile, idx)
    test_pred, test_err = testing(test_data, model)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_regression_results(ax, test_data.loc[:,["BicepC"]].values, test_pred,'LinearRegression test','MSE={:.2f} cm'.format(test_err),"BicepC")
    plt.tight_layout()
    plt.show()
    # pass
