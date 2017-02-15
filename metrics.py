"""
	Predicts test data by creating models and
	Creates the performance metrics for each of the models present in model module
"""

import itertools
import os
from time import time
import model
import pre_process
from sklearn import metrics
import matplotlib.pyplot as plt

if __name__ == "__main__":
	AVAILABLE_METRICS = os.listdir("metrics/")
	for mod in model.MODELS:
		if mod not in AVAILABLE_METRICS:
			os.makedirs("metrics/"+mod)
			print("processing " + mod)
			with open("metrics/" + mod + "/" + "timings.txt", "w") as timings:

				print("loading model")
				T0 = time()
				classifier = model.load(mod)
				timings.write("training time: " + str(round(time()-T0, 3)) + "s")
				timings.write("\n")
				print("loading test data")
				test_data, labels = pre_process.create_test_data()
				print("predicting test data")
				T0 = time()
				predicted = classifier.predict(test_data.values)
				timings.write("Prediction time: " + str(round(time()-T0, 3)) + "s")
				timings.write("\n")

			print("creating metrics")
			cm = metrics.confusion_matrix(labels, predicted)
			plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
			plt.title("Confusion Matrix")
			plt.colorbar()
			thresh = cm.max() / 2
			for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
				plt.text(j, i, cm[i, j], horizontalalignment="center",
													color="white" if cm[i, j] > thresh else "black")
			plt.tight_layout()
			plt.ylabel('True label')
			plt.xlabel('Predicted label')
			plt.savefig("metrics/" + mod + "/" + "confusion_matrix.png")
			plt.gcf().clear()
			with open("metrics/" + mod + "/" + "classification_report.txt", "w") as report:
				report.write("Classification report for classifier %s:\n%s\n"
				      							% (classifier, metrics.classification_report(labels, predicted)))
				report.write("\n")
