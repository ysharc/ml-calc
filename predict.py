import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.externals import joblib
import os

def train():
	train = pd.read_csv("train_data/train.csv")
	#test  = pd.read_csv("../input/test.csv")

	train_labels = train.pop("label")

	knn = KNeighborsClassifier(n_neighbors=1)
	knn.fit(train, train_labels)

	#predicted = knn.predict(test)
	joblib.dump(knn, "digit_recognition.pkl")

def predict(number_recognizer, image):
	return number_recognizer.predict(image)

def load():
	return joblib.load("digit_recognition.pkl")

if __name__ == "__main__":
	if "digit_recognition.pkl" not in os.listdir():
		train()
	number_recognizer = joblib.load("digit_recognition.pkl")