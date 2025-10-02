import cv2
import numpy as np


def process_image(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    _, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    return thresholded
