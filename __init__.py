import cv2
import numpy as np
import predict

for i in range(10):
	test_img = cv2.imread('test_images/'+str(i)+'.png')

	gray_scale_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

	ret, threshold = cv2.threshold(gray_scale_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	predict_model = predict.load()

	def handle_shape(row_min, row_max, col_min, col_max):
		row_diff = max(0, 28 - row_max + row_min)
		col_diff = max(0, 28 - col_max + col_min)
		if row_diff > 0:
			row_left_padding = int(row_diff/2)
			row_right_padding = int(row_diff/2)
			if (row_diff/2):
				row_left_padding += 1
		if col_diff > 0:
			col_left_padding = int(col_diff/2)
			col_right_padding = int(col_diff/2)
			if (col_diff/2):
				col_left_padding += 1
		
		return row_left_padding, row_right_padding, col_left_padding, col_right_padding

	ret, markers = cv2.connectedComponents(threshold)
	markers = markers + 1

	markers = cv2.watershed(test_img, markers)

	test_img[markers == -1] = 	[0, 0, 0]

	total = 0
	correct = 0
	wrong = {}

	print(np.unique(markers).size)
	for group in np.unique(markers):
		if group not in [1, -1]:
			columns, rows = np.where(markers == group)
			row_min = min(rows)
			row_max = max(rows)
			column_min = min(columns)
			column_max = max(columns)
			if 15 < (column_max- column_min) * (row_max - row_min) < 200:
				a,b,c,d = handle_shape(row_min, row_max, column_min, column_max)
				row_min -= 4
				row_max += 4
				column_min -= 4
				column_max += 4
				#print(row_min, row_max, column_min, column_max)
				if row_min < 0:
					row_max -= row_min
					row_min = 0
				if column_min < 0:
					column_max -= column_min
					column_min = 0
				temp_img = gray_scale_img[column_min:column_max, row_min:row_max]
				resized = cv2.resize(temp_img, (28,28))
				resized = (255 - resized)
				#cv2.imwrite("temp/" + str(i) + str(group)+".jpg", resized)
				flattened = resized.flatten()
				total += 1
				#print(predict_model.predict(flattened.reshape(1, -1)))
				t = predict_model.predict(flattened.reshape(1, -1))[0]
				if predict_model.predict(flattened.reshape(1,-1))[0] == i:
					correct += 1
				else:
					if t in wrong:
						wrong[t] += 1
					else:
						wrong[t] = 1

	print(wrong)
	print((correct/total)*100)