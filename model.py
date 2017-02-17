"""
	Used for training and loading models for the character recognition.
"""

import os
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
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

def knn_1_neighbor():
	"""
		nearest neighbor classification
	"""
	knn = KNeighborsClassifier(n_neighbors=1)
	knn.fit(TRAIN_DATA, LABELS)
	return knn

def knn_10_neighbors():
	"""
		knn with 10 neighbors
	"""
	knn = KNeighborsClassifier(n_neighbors=10)
	knn.fit(TRAIN_DATA, LABELS)
	return knn

def svm_default():
	"""
		SVM with deafult parameters
	"""
	svm_classifier = SVC()
	svm_classifier.fit(TRAIN_DATA, LABELS)
	return svm_classifier

def ada_boost_default():
	"""
		ada_boost on decision trees with deafult parameters
	"""
	ab_classifier = AdaBoostClassifier(n_estimators=600)
	ab_classifier.fit(TRAIN_DATA, LABELS)
	return ab_classifier

def random_forest_classifier():
	"""
		Random Forest with default parameters
	"""
	rf_classifier = RandomForestClassifier(n_estimators=100)
	rf_classifier.fit(TRAIN_DATA, LABELS)
	return rf_classifier

MODELS = {"knn_default": knn_default,
										"knn_1_neighbor": knn_1_neighbor,
										"knn_10_neighbors": knn_10_neighbors,
										"svm_default": svm_default,
										"ada_boost": ada_boost_default,
										"random_forest": random_forest_classifier}
TRAIN_DATA = pd.read_csv("train_data/train.csv")
LABELS = TRAIN_DATA.pop("label")
