from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
from loguru import logger

from src.imip import IMIP, MockIMIP

imip = MockIMIP()  # Replace with actual IMIP instance when running in production


class ProductAnalyzer:
    def analyze_patch(
        self,
        image: np.ndarray,
    ) -> None:
        line_blue, line_red = self.apply_shadow_detection_algorithm(image, 2, 3)

    def apply_shadow_detection_algorithm(
        self, image: np.ndarray, red_multiplier: int, blue_multiplier: int
    ) -> tuple[tuple[Any, Any] | None, tuple[Any, Any] | None]:
        img_diff = self.get_red_and_blue_channel_difference(
            image, red_multiplier, blue_multiplier
        )

        img_sobel = cv.Sobel(img_diff, cv.CV_32F, 0, 1, ksize=5)  # Vertical gradient
        abs_sobel = np.abs(img_sobel)
        if abs_sobel.max() == 0:
            logger.warning("Sobel filter resulted in an image with all zero values")
            return None, None
        img_scaled_diff = (255 * abs_sobel / np.max(abs_sobel)).astype(np.uint8)

        img_scaled_diff = cv.bilateralFilter(img_scaled_diff, 9, 75, 75)

        # IMIP
        imip.debug(img_scaled_diff, "Scaled Sobel Image")

        img_thresh = cv.adaptiveThreshold(
            img_scaled_diff, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, -30
        )

        # IMIP
        imip.debug(img_thresh, "Adaptive Threshold Image")

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        img_shadows = cv.morphologyEx(img_thresh, cv.MORPH_CLOSE, kernel, iterations=4)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
        img_shadows = cv.morphologyEx(
            img_shadows, cv.MORPH_CLOSE, kernel, iterations=10
        )
        img_shadows = cv.morphologyEx(img_shadows, cv.MORPH_OPEN, kernel, iterations=30)

        img_eroded = cv.erode(img_shadows, kernel, iterations=2)

        # IMIP
        imip.debug(img_eroded, "Eroded Shadow Image")

        numLabels, labels, stats, centroids = cv.connectedComponentsWithStats(
            img_eroded, connectivity=8
        )

        numlabel_top = 0
        numlabel_bottom = 0
        min_area = 200
        max_area = 10_000
        for i in range(1, numLabels):
            area = stats[i][4]
            if area < min_area or area > max_area:
                continue
            _, cy = centroids[i]

            if cy <= centroids[numlabel_top][1]:
                if numlabel_top == 0 or area >= stats[numlabel_top][4]:
                    numlabel_top = i
            if cy > centroids[numlabel_bottom][1]:
                if numlabel_bottom == 0 or area >= stats[numlabel_bottom][4]:
                    numlabel_bottom = i

        line_blue = (
            self.get_shadow_end_points(numlabel_top, labels, find_max=True)
            if numlabel_top != 0
            else None
        )
        line_red = (
            self.get_shadow_end_points(numlabel_bottom, labels, find_max=False)
            if numlabel_bottom != 0
            else None
        )

        return line_blue, line_red

    @staticmethod
    def get_red_and_blue_channel_difference(
        image: np.ndarray, red_multiplier: int, blue_multiplier: int
    ) -> np.ndarray:
        if len(image.shape) < 3 or image.shape[2] < 3:
            raise ValueError(
                "Input image must be a color image with at least 3 channels"
            )
        blue, _, red = cv.split(image)

        red = cv.bilateralFilter(red, 9, 75, 75)
        blue = cv.bilateralFilter(blue, 9, 75, 75)

        red_dark = cv.subtract(red, np.ones_like(red) * red_multiplier)
        blue_dark = cv.subtract(blue, np.ones_like(blue) * blue_multiplier)

        red_negative = red - blue_dark
        blue_negative = blue - red_dark

        return np.maximum(red_negative, blue_negative)

    @staticmethod
    def get_shadow_end_points(
        num_label: int,
        labels: np.ndarray,
        find_max: bool,
    ) -> tuple[Any, Any] | None:
        line = []
        for col in range(labels.shape[1]):
            # Only consider pixels belonging to the current label
            ys = np.where(labels[:, col] == num_label)[0]
            if ys.size > 0:
                if find_max:
                    y_value = ys.max()
                else:
                    y_value = ys.min()
                line.append((col, y_value))

        if len(line) == 0:
            return None

        # Convert to numpy array for polyfit
        line_np = np.array(line)
        # Fit a line (y = m*x + b) through the points
        fit = np.polyfit(line_np[:, 0], line_np[:, 1], 1)
        m, b = fit
        x0, x1 = line_np[0, 0], line_np[-1, 0]
        y0 = int(m * x0 + b)
        y1 = int(m * x1 + b)
        return ({"x": int(x0), "y": y0}, {"x": int(x1), "y": y1})


if __name__ == "__main__":
    product_analyzer = ProductAnalyzer()

    # Dev
    imip = IMIP(
        images=Path(
            "C:\\Users\\joeri\\Documents\\IMA\\images\\2026-03-06\\breed\\crops\\kleine_patch"
        )
    )
    imip.debug_fn(
        product_analyzer.apply_shadow_detection_algorithm,
        watch_file=Path(__file__),
        red_multiplier=50,
        blue_multiplier=10,
    )
