# library
import numpy as np
import cv2
from matplotlib import pyplot as plt

# desired input
img = cv2.imread(r'C:\Users\dell\Desktop\road_damage\Input-Set\Carcked_01.jpg')

# check if image is loaded
if img is None:
    print("Image not found. Check the file path.")
    exit()

# grayscale + blur
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray, (3, 3))

# Logarithmic transformation
img_log = (np.log(blur + 1) / (np.log(1 + np.max(blur)))) * 255
img_log = np.array(img_log, dtype=np.uint8)

# Bilateral filtering
bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)

# Edge detection
edges = cv2.Canny(bilateral, 100, 200)

# Morphological closing
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Feature detection using ORB
orb = cv2.ORB_create(nfeatures=1500)
keypoints, descriptors = orb.detectAndCompute(closing, None)
featuredImg = cv2.drawKeypoints(closing, keypoints, None)

# Output
cv2.imshow("Road Crack Detected", featuredImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
