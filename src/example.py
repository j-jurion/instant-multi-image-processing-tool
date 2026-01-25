from pathlib import Path

import cv2 as cv
import numpy as np

from imip import IMIP


def process_image(image: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    imip.debugger(gray, "Gray Image")
    blurred = cv.GaussianBlur(image, (5, 5), 2)
    imip.debugger(blurred, "Blurred Image")
    _, thresholded = cv.threshold(blurred, 50, 255, cv.THRESH_BINARY)
    imip.debugger(thresholded, "Thresholded Image")
    return thresholded


def analyse_image(image: np.ndarray) -> int:
    processed_image = process_image(image)
    contours, _ = cv.findContours(
        processed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    return len(contours)


if __name__ == "__main__":
    imip = IMIP()
    imip.set_debug_save_dir(output_directory=Path("./debug_images"))
    image_folder = Path("./images")
    imip.load_images(image_folder)
    imip.debug_fn(process_image, watch_file=Path("src/example.py"))
