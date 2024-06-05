from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import svm
from sklearn.model_selection import train_test_split

def process(path, app, model_path):
	MODELS_FOLDER = os.getenv("MODELS_FOLDER")
	app.config["MODELS_FOLDER"] = MODELS_FOLDER

	X=pd.read_csv(path, usecols=range(40), header=None)	
	y=pd.read_csv(path,usecols=[40],header=None)
	X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.3, random_state=42)
	svmlk_model=svm.SVC(kernel ='linear', C = 1, probability=True)
	svmlk_model.fit(X_train, y_train)
	joblib.dump(svmlk_model, model_path)
	# joblib.dump(svmlk_model, 'svmlk_model.joblib')
	y_pred = svmlk_model.predict(X_test)
	print("predicted")

	r2 = round(r2_score(y_test, y_pred) * 100, 2)
	ac = round(accuracy_score(y_test,y_pred) * 100, 2)
	mse = round(mean_squared_error(y_test, y_pred) * 100, 2)
	mae = round(mean_absolute_error(y_test, y_pred) * 100, 2)
	rms = round(np.sqrt(mean_squared_error(y_test, y_pred)) * 100, 2)

	return mse, mae, r2, rms, ac
	