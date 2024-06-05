import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def process(path, app, model_path):
    MODELS_FOLDER = os.getenv("MODELS_FOLDER")
    app.config["MODELS_FOLDER"] = MODELS_FOLDER

    X = pd.read_csv(path, usecols=range(40), header=None)
    y = pd.read_csv(path, usecols=[40], header=None)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.values.ravel(), test_size=0.3, random_state=42
    )
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, model_path)
    # joblib.dump(rf_model, 'rf_model.joblib')
    y_pred = rf_model.predict(X_test)
    print("predicted")

    r2 = round(r2_score(y_test, y_pred) * 100, 2)
    ac = round(accuracy_score(y_test, y_pred) * 100, 2)
    mse = round(mean_squared_error(y_test, y_pred) * 100, 2)
    mae = round(mean_absolute_error(y_test, y_pred) * 100, 2)
    rms = round(np.sqrt(mean_squared_error(y_test, y_pred)) * 100, 2)

    return mse, mae, r2, rms, ac
