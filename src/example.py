import cv2 as cv
import numpy as np

from imip import IMIP


def process_image(image: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    imip.debugger(gray, "Gray Image")
    blurred = cv.GaussianBlur(gray, (5, 5), 2)
    imip.debugger(blurred, "Blurred Image")
    _, thresholded = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY)
    imip.debugger(thresholded, "Thresholded Image")
    return thresholded

def analyse_image(image: np.ndarray) -> int:
    processed_image = process_image(image)
    contours, _ = cv.findContours(processed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return len(contours)


if __name__ == "__main__":
    # # Easy one-image example
    # image = cv.imread("./images/001.jpg")
    # if image is not None:
    #     processed_image = process_image(image)
    # else:
    #     print("Error: Image could not be loaded.")
    # imip.show_debug_images()

    # multiple-image example
    imip = IMIP()
    imip.set_debugger_options(output_directory="./debug_images", debug_images_saved=True)
    image_folder = "./images"
    imip.load(image_folder)
    imip.debug_fn(analyse_image, watch_file="./src/example.py")


