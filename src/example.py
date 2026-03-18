from pathlib import Path

import cv2 as cv
import numpy as np

from imip import IMIP


def process_image(image: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    imip.debug(gray, "Gray Image")
    blurred = cv.GaussianBlur(gray, (5, 5), 4)
    imip.debug(blurred, "Blurred Image")
    _, thresholded = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY)
    imip.debug(thresholded, "Thresholded Image")
    return thresholded


def analyse_image(image: np.ndarray) -> int:
    processed_image = process_image(image)
    contours, _ = cv.findContours(
        processed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    return len(contours)


if __name__ == "__main__":
    image_folder = Path("./images")
    imip = IMIP(
        images=image_folder,
        output_directory=Path("./debug_images"),
        save_debug_images=True,
    )
    imip.debug_fn(process_image, watch_file=Path(__file__))
