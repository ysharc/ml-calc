"""
	Used for training and loading models for the character recognition.
"""

import os
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def load(model_name, parameters=None):
	"""
		returns the model the user wants to use.
		if model is not already available in the models, it is trained and returned
	"""
	if model_name + '.pkl' in os.listdir('models') and not parameters:
		return joblib.load('models/' + model_name + '.pkl')

	model = MODELS[model_name]()
	joblib.dump(model, 'models/' +  model_name + ".pkl")
	return model

def knn_default():
	"""
		the knn classifier with all default parameters
	"""
	knn = KNeighborsClassifier()
	knn.fit(TRAIN_DATA, LABELS)
	return knn

MODELS = {"knn_default": knn_default}
TRAIN_DATA = pd.read_csv("train_data/train.csv")
LABELS = TRAIN_DATA.pop("label")
