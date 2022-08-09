import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def training(train_data: float):
	"""train linear regression model on training dataset"""
	X = train_data.loc[:,["height","weight","age"]]
	y = train_data.loc[:,["BicepC"]]
	print("Dataset shape\n Feature Matrix:{} Target Vector:{}"\
	  .format(X.shape,y.shape))

	model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
	model.fit(X,y)
	print("Parameters M: {}".format(model.coef_))
	print("Parameters b: {}".format(model.intercept_))

	y_pred = model.predict(X)
	err = mean_squared_error(y,y_pred)
	print("Mean Squared Error: {}".format(err))

	return y_pred, err, model