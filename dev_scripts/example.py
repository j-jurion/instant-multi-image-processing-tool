from pathlib import Path

import cv2 as cv
import numpy as np
import typer

from src.imip import IMIP

app = typer.Typer()


def process_image(image: np.ndarray, threshold: int) -> np.ndarray:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    imip.debug(gray, "Gray Image")
    blurred = cv.GaussianBlur(gray, (5, 5), 4)
    imip.debug(blurred, "Blurred Image")
    _, thresholded = cv.threshold(blurred, threshold, 255, cv.THRESH_BINARY)
    imip.debug(thresholded, "Thresholded Image")
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
    app()
