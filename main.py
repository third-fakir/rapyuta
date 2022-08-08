# import library you made (regression) and 3rd party library here
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from regression.plot import plot_regression_results

if __name__ == "__main__":
    # implement your application code here
    filename = "../data/regression_train.csv"
    train_data = pd.read_csv(filename,index_col='ID')
    train_data.sample(2)
    train_stats = train_data.describe().transpose()
    print(train_stats)

    scatter_matrix(train_data, alpha=0.5, figsize=(10, 10), diagonal='hist');
    train_data_n = (train_data - train_data.mean()) /train_data.std()
    train_data_n.describe().transpose()
    train_data_n.plot.box()

    X = train_data.loc[:,["height","weight","age"]]
    y = train_data.loc[:,["BicepC"]]
    print("Dataset shape\n Feature Matrix:{} Target Vector:{}"\
      .format(X.shape,y.shape))

    model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
    model.fit(X,y)
    print("Parameters M: {}".format(model.coef_))
    print("Parameters b: {}".format(model.intercept_))

    y_pred = model.predict(X)
    print("Mean Squared Error: {}".format(mean_squared_error(y,y_pred)))

    fig, ax = plt.subplots(figsize=(8, 8))
    y_pred = model.predict(X)
    plot_regression_results(ax, y.values, y_pred,'LinearRegression','MSE={:.2f} cm'.format(mean_squared_error(y,y_pred)),"BicepC")
    plt.tight_layout()
    plt.show()

    test_data = pd.read_csv("../data/regression_test.csv", index_col='ID')
    test_data.sample(2)
    test_stats = test_data.describe().transpose()
    print(test_stats)
    test_data_n = (test_data - test_data.mean()) /test_data.std()
    test_data_n.describe().transpose()
    test_data_n.plot.box()

    X_test = test_data.loc[:,["height","weight","age"]]
    y_test = test_data.loc[:,["BicepC"]]
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_regression_results(ax, y_test.values, y_pred,'LinearRegression test','MSE={:.2f} cm'.format(mean_squared_error(y_test,y_pred)),"BicepC")
    plt.tight_layout()
    plt.show()
    pass
