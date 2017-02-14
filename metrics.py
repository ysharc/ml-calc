"""
	Predicts test data by creating models and
	Creates the performance metrics for each of the models present in model module
"""

import os
import model
import pre_process
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

if __name__ == "__main__":
	AVAILABLE_METRICS = os.listdir("metrics/")
	for mod in model.MODELS:
		if mod not in AVAILABLE_METRICS:
			os.makedirs("metrics/"+mod)
			print("loading model")
			classifier = model.load(mod)
			print("loading test data")
			test_data, labels = pre_process.create_test_data()
			print("predicting test data")
			predicted = classifier.predict(test_data.values)
			print("creating metrics")
			actual_vs_predicted = {i: {} for i in range(10)}
			for i in range(len(labels)):
				if predicted[i] in actual_vs_predicted[labels[i]]:
					actual_vs_predicted[labels[i]][predicted[i]] += 1
				else:
					actual_vs_predicted[labels[i]][predicted[i]] = 1
			print(accuracy_score(labels, predicted))
			print("saving images")
			for number in actual_vs_predicted:
				plt.bar(list(actual_vs_predicted[number].keys()), list(actual_vs_predicted[number].values()))
				plt.savefig("metrics/" + mod + "/" + str(number)+".png")
				plt.gcf().clear()
