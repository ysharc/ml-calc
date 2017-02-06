import cv2
import numpy as np

test_img = cv2.imread('test.jpg')

gray_scale_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

ret, threshold = cv2.threshold(gray_scale_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
'''
#noise removal
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations = 1)

#sure background
sure_background = cv2.dilate(opening, kernel, iterations=1)

#Finding sure foreground
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_foreground = cv2.threshold(dist_transform, 0.9*dist_transform.max(), 255, 0)

#Finding unknown region
sure_foreground = np.uint8(sure_foreground)
unknown = cv2.subtract(sure_background, sure_foreground)
'''
ret, markers = cv2.connectedComponents(threshold)
markers = markers + 1

markers = cv2.watershed(test_img, markers)

test_img[markers == -1] = 	[0, 0, 0]

for group in np.unique(markers):
	if group not in [1, -1]:
		columns, rows = np.where(markers == group)
		row_min = min(rows)
		row_max = max(rows)
		column_min = min(columns)
		column_max = max(columns)
		cv2.imwrite(str(group)+".jpg", test_img[column_min:column_max, row_min:row_max])