from pathlib import Path

import cv2 as cv
import numpy as np
import typer

# Enable auto-visualization when imip_images variable is created
from dev_scripts.my_visualizer import (
    enable_auto_visualization_on_variable,
    show_accumulated,
)
from src.imip import IMIP

enable_auto_visualization_on_variable("imip_images")

app = typer.Typer()


def process_image(image: np.ndarray, threshold: int) -> np.ndarray:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 4)
    _, thresholded = cv.threshold(blurred, 50, 255, cv.THRESH_BINARY)
    imip_images = [image, gray, blurred, thresholded]  # noqa: F841
    return thresholded


@app.command()
def main(
    image_folder: Path = typer.Argument(
        ..., help="Path to the folder containing images to process"
    ),
    output_directory: Path = typer.Option(
        None, help="Path to the output directory for debug images"
    ),
):
    global imip
    imip = IMIP(
        images=image_folder,
        output_directory=output_directory,
    )
    imip.debug_fn(process_image, watch_file=Path(__file__), threshold=60)


if __name__ == "__main__":
    # imip = IMIP(
    #     images=Path(
    #         "C:\\Users\\joeri\\Documents\\IMA\\images\\2026-03-06\\breed\\crops\\kleine_patch"
    #     )
    # )

    image = cv.imread(r"C:\Users\joeri\Pictures\pretty images\04.jpg")
    assert image is not None, "Failed to load image. Please check the path."

    image2 = cv.imread(r"C:\Users\joeri\Pictures\pretty images\03.jpg")
    assert image2 is not None, "Failed to load image. Please check the path."

    process_image(image, threshold=60)
    process_image(image2, threshold=60)
    show_accumulated()
