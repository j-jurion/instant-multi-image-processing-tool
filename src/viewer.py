import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from base import ImageData, ProcessedImageData


class Viewer:
    def __init__(self, original_images: list[ImageData]):
        self.original_images = original_images
        self.processed_images: list[ProcessedImageData] = []
        self.scale = 1.0
        self.columns = 3
        plt.ion()
        self.fig, self.ax = plt.subplots()
        plt.axis("off")
        self.im = None

    def reformat_image(
        self, new_size: tuple[int, int], image: np.ndarray
    ) -> np.ndarray:
        if len(image.shape) == 2:  # Grayscale image
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:  # BGR image
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        else:
            raise ValueError("Unsupported image format")
        return cv.resize(image, new_size)

    def make_grid(self):
        if not self.original_images:
            return None
        h, w = self.original_images[0].image.shape[:2]
        h, w = int(h * self.scale), int(w * self.scale)

        if self.processed_images:
            reformatted = [
                self.reformat_image((w, h), img.image) for img in self.processed_images
            ]
        else:
            reformatted = [
                self.reformat_image((w, h), img.image) for img in self.original_images
            ]

        while len(reformatted) % self.columns != 0:
            reformatted.append(np.zeros_like(reformatted[0]))

        rows = []
        for i in range(0, len(reformatted), self.columns):
            row = np.hstack(reformatted[i : i + self.columns])
            rows.append(row)

        return np.vstack(rows)

    def show(self):
        grid = self.make_grid()
        if grid is not None:
            if self.im is None:
                self.im = self.ax.imshow(grid)
            else:
                self.im.set_data(grid)

    def update(self, processed_images: list[ProcessedImageData]):
        self.processed_images = processed_images
        self.show()
