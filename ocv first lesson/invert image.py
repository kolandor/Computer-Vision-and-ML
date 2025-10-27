import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

img = cv2.imread('test_img.png')

if img is None:
	raise SystemExit("Could not read 'test_img.png'. Make sure the file exists in the script directory.")

try:
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	plt.imshow(img_rgb)
	plt.show()
except Exception as e:
	print('Matplotlib display failed:', e)
