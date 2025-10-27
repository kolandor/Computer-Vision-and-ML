import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

img = cv2.imread('test_img.png')

if img is None:
	raise SystemExit("Could not read 'test_img.png'. Make sure the file exists in the script directory.")

# Prefer matplotlib display (convert BGR -> RGB for correct colors)
try:
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	plt.imshow(img_rgb)
	plt.axis('off')
	plt.show()  # this blocks until the window is closed
except Exception as e:
	print('Matplotlib display failed:', e)

# Fallback to OpenCV window (blocks until a key is pressed)
try:
	cv2.imshow('Image (OpenCV)', img)
	print('Focus the image window and press any key to close it.')
	cv2.waitKey(0)
	cv2.destroyAllWindows()
except Exception as e:
	print('OpenCV display failed:', e)
	input('Press Enter to exit...')

