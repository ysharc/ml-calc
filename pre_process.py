"""
	Creates a csv file from the test images, for the models to use as a test set
"""

import os
import csv
import pandas as pd
import cv2
import numpy as np

def create_test_data():
	"""
		Returns test_data and labels in the form of numpy ndarray
	"""
	if "test_data.csv" not in os.listdir("test_data/"):
		with open("test_data/test_data.csv", "w", newline="") as test_data:
			csv_writer = csv.writer(test_data)
			csv_writer.writerow(["label"] + ["pixel" + str(i) for i in range(784)])
			for instance in os.listdir("test_images"):
				test_img = cv2.imread('test_images/'+instance)

				gray_scale_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

				ret, threshold = cv2.threshold(gray_scale_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

				ret, markers = cv2.connectedComponents(threshold)
				markers = markers + 1

				markers = cv2.watershed(test_img, markers)

				test_img[markers == -1] = [0, 0, 0]
				for group in np.unique(markers):
					row = [instance[0]]
					if group not in [1, -1]:
						columns, rows = np.where(markers == group)
						row_min = min(rows)
						row_max = max(rows)
						column_min = min(columns)
						column_max = max(columns)
						if 15 < (column_max- column_min) * (row_max - row_min) < 200:
							row_min -= 4
							row_max += 4
							column_min -= 4
							column_max += 4
							if row_min < 0:
								row_max -= row_min
								row_min = 0
							if column_min < 0:
								column_max -= column_min
								column_min = 0
							temp_img = gray_scale_img[column_min:column_max, row_min:row_max]
							resized = cv2.resize(temp_img, (28, 28))
							resized = (255 - resized)
							flattened = resized.flatten()
							row += flattened.tolist()
							csv_writer.writerow(row)
	test_data = pd.read_csv("test_data/test_data.csv")
	labels = test_data.pop("label")
	return test_data, labels

if __name__ == "__main__":
	create_test_data()