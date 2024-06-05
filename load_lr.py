import joblib
import pandas as pd
import numpy as np
import librosa

SAMPLE_RATE = 44100

def extract_features(path):
	b, _ = librosa.core.load(path, sr = SAMPLE_RATE)
	assert _ == SAMPLE_RATE
	try:
		gmm = librosa.feature.mfcc(y=b, sr = SAMPLE_RATE, n_mfcc=20)
		spectral_centroids = librosa.feature.spectral_centroid(y=b, sr=SAMPLE_RATE)[0]
		return pd.Series(np.hstack((np.mean(gmm, axis=1), np.std(gmm, axis=1))))
	except Exception as e:
		print("Error processing file:", e)
		print('bad file')
		return pd.Series([0]*40)

def process(audio_file_path, model_path="lr_model.joblib"):
    # Load the saved logistic regression model
    model = joblib.load(model_path)

    try:
        features = extract_features(audio_file_path)
        X_test = [features]
        y_pred = model.predict(X_test)

        result = ""
        if y_pred[0] == 1:
            probabilities = model.predict_proba(X_test)[0]
            depression_probability = probabilities[1]
            result = f"Depressed: {depression_probability * 100:.2f}%"
        else:
            probabilities = model.predict_proba(X_test)[0]
            normal_probability = probabilities[0]
            result = f"Normal: {normal_probability * 100:.2f}%"

        return result
    except Exception as e:
        print("Error predicting:", e)