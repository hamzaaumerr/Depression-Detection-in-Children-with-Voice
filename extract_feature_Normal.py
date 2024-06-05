import pandas as pd
import numpy as np
import os
import csv
import librosa
import glob

SAMPLE_RATE = 44100
# Returns MFCC features with mean and standard deviation along time
def get_mfcc(name, path):
    b, _ = librosa.core.load(path + name, sr=SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    try:
        gmm = librosa.feature.mfcc(y=b, sr=SAMPLE_RATE, n_mfcc=20)
        return pd.Series(np.hstack((np.mean(gmm, axis=1), np.std(gmm, axis=1))))
    except Exception as e:
        print('An exception occurred:', str(e))
        print('bad file')
        return pd.Series([0] * 40)

def process(path):
	mfcc=[]
	name=[]
	for i in range(1,2):
		cl=0
		for file in glob.glob(path+"/*.wav".format(i)):
			name.append(file.split('\\')[-1].split('.')[0])
			filename = file.split('\\')[-1].split('.')[0]+".wav"
			s=get_mfcc(filename, path+'/')
			mfcc.append(s.tolist())
	full=[]
	for i in range(0,len(mfcc)):
		d=[]
		for j in mfcc[i]:
			d.append(j)
		d.append(0)
		d.append(name[i])

		full.append(d)
	# Specify the output directory directly
	output_dir = './results'
	os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    # Save the CSV file in the output directory
	csv_path = os.path.join(output_dir, 'NormalMFCC.csv')
	with open(csv_path, 'w',newline='') as myfile:
		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		for x in full:
			wr.writerow(x)