'''
Purpose: visualize data from the dataframe
'''
import os
import cv2
import pandas as pd

df = pd.read_csv("distance-estimator/data/predictions.csv")

for idx, row in df.iterrows():
	if os.path.exists(os.path.join("original_data/train_annots/", row['filename'])):
		fp = os.path.join("original_data/train_images", row['filename'].replace('.txt', '.png'))
		im = cv2.imread(fp)

		x1 = int(row['xmin'])
		y1 = int(row['ymin'])
		x2 = int(row['xmax'])
		y2 = int(row['ymax'])

		cv2.line(im, (int(1224/2), 0), (int(1224/2), 370), (255,255,255), 2)
		cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3)
		string = "(act: {}, pred: {})".format(row['zloc'], row['zloc_pred'])
		cv2.putText(im, string, (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

		cv2.imshow("detections", im)
		cv2.waitKey(0)
