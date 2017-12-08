#!/usr/bin/python2
import compute
import cv2
import sys
import time
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def validate(rect, ww, hh):
	x, y, w, h = rect
	return (x > w) and (y > h) and (x + w*2 < ww) and (y + h*2 < hh)

if __name__ == "__main__":
	filename = sys.argv[1]
	with open(filename.replace('jpg', 'json'), 'r') as f:
		markup = json.load(f)
		rect = markup['rect']
		name = markup['name']
	image = cv2.imread(filename)
	hh, ww = image.shape[:2]
	if not validate(rect, ww, hh):
		print('invalid rect: {} om=n image ({} {})'.format(rect, w, h))

	step = 5
	x, y, w, h = rect
	det = compute.YoloDetector()
	hres, wres = (h+step-1)/step, (w+step-1)/step

	result = np.full((hres, wres), 0.0, dtype=np.float)
	video = cv2.VideoWriter("test.avi", cv2.VideoWriter_fourcc(*'MJPG'), 60., (w*2, h*2), True)
	start = time.time()
	for j in xrange(x, x+w, step):
		for i in xrange(y, y+h, step):
			ii, jj = (i-y)/step, (j-x)/step
			im = image[i-h:i+h, j-w:j+w, :]
			result[ii][jj] = det.get_class_confidence(im.copy(), name)
			video.write(im)
	print('Elapsed time: {} seconds'.format(time.time() - start))
	video.release()

	sns.heatmap(result)
	plt.show()