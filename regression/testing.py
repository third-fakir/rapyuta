import pandas as pd
from sklearn.metrics import mean_squared_error

def testing(test_data: float, model: object):
	"""test the model on test data, and return predicted values, and MSE"""

	test_data_n = (test_data - test_data.mean()) /test_data.std()
	test_data_n.describe().transpose()
	test_data_n.plot.box()

	X_test = test_data.loc[:,["height","weight","age"]]
	y_test = test_data.loc[:,["BicepC"]]
	y_pred = model.predict(X_test)

	return y_pred, mean_squared_error(y_test,y_pred)