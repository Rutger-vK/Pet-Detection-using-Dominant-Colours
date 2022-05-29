# https://pysource.com/instance-segmentation-mask-rcnn-with-python-and-opencv
import cv2
import numpy as np
from copy import copy


def remove_background(img):
	img_original = copy(img)
	# Loading Mask RCNN
	net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
										"dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

	height, width, _ = img.shape

	# Create black image
	black_image = np.zeros((height, width, 3), np.uint8)
	black_image[:] = (0, 0, 0)

	# Detect objects
	blob = cv2.dnn.blobFromImage(img, swapRB=True)
	net.setInput(blob)

	boxes, masks = net.forward(["detection_out_final", "detection_masks"])
	detection_count = boxes.shape[2]

	for i in range(detection_count):
		box = boxes[0, 0, i]
		class_id = box[1]
		score = box[2]
		if score < 0.5:
			continue

		# Get box Coordinates
		x = int(box[3] * width)
		y = int(box[4] * height)
		x2 = int(box[5] * width)
		y2 = int(box[6] * height)

		roi = black_image[y: y2, x: x2]
		roi_height, roi_width, _ = roi.shape

		# Get the mask
		mask = masks[i, int(class_id)]
		mask = cv2.resize(mask, (roi_width, roi_height))
		_, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

		# Get mask coordinates
		contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			cv2.fillPoly(roi, [cnt], (255, 255, 255))

	lower = np.array([0, 0, 0])
	upper = np.array([5, 5, 5])

	# Create mask to only select black
	thresh = cv2.inRange(black_image, lower, upper)

	# apply morphology
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
	morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	mask = 255 - thresh

	final = cv2.bitwise_and(img_original, img_original, mask=mask)

	return final
