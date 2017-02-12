"""
	Used for training and loading models for the character recognition.
"""

import os
from sklearn.externals import joblib

MODELS = {}

def load(model_name, parameters=None):
	"""
		returns the model the user wants to use.
		if model is not already available in the models, it is trained and returned
	"""
	if model_name + '.pkl' in os.listdir('models') and not parameters:
		return joblib.load(model_name + '.pkl')

	model = MODELS[model_name]()
	joblib.dump(model, model_name + ".pkl")
	return model
