import cv2
import numpy as np

from src.main import ProcessingResult


def process_image(processed: ProcessingResult, img: np.ndarray):
    processing_result = ProcessingResult([], [])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    processed.append_image(image=blurred, description="Blurred Image")
    _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    processed.append_image(image=thresholded, description="Thresholded Image")
    processed.append_data(data=np.mean(blurred), description="Mean Intensity")

    return processing_result
