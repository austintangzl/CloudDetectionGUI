import cv2
import numpy as np
from matplotlib import pyplot as plt

filename = "C:\CloudDetectionGUI3.6\For Brenda\T1_00006197\T012017101306250892A_L2.tif"

image = cv2.imread(filename)

edge = cv2.Canny(image, 60, 120)

output = "C:\CloudDetectionGUI3.6\For Brenda\T1_00006197\T012017101306250892A_L2_EdgyTeen.tif"

cv2.imwrite(output, edge)

# cv2.imshow('edgy', edge)
# cv2.waitKey()
# cv2.destroyAllWindows()


