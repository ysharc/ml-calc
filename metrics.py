"""
	Predicts test data by creating models and
	Creates the performance metrics for each of the models present in model module
"""

import os
import model
import pre_process

if __name__ == "__main__":
	AVAILABLE_METRICS = os.listdir("metrics/")
	for mod in model.MODELS:
		if mod not in AVAILABLE_METRICS:
			classifier = model.load(mod)
			test_data, labels = pre_process.create_test_data()
			predicted = classifier.predict(test_data)
