import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from .base import ImageBundle

LABEL_WIDTH = 150
LABEL_HEIGHT = 40
FONT_SCALE = 0.5
FONT_THICKNESS = 1
DEFAULT_SCALE = 1.0
DEFAULT_COLUMNS = 4
PADDING_COLOR = 255


class Viewer:
    def __init__(self, image_bundles: list[ImageBundle]):
        self.image_bundles = image_bundles
        self.scale = DEFAULT_SCALE
        self.columns = DEFAULT_COLUMNS
        self._setup_matplotlib()

    def _setup_matplotlib(self) -> None:
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.axis("off")
        self.im = None

    def get_columns(self, image_bundles: list[ImageBundle]) -> int:
        if not image_bundles:
            return 0
        return max(1 + len(bundle.processed_images) for bundle in image_bundles)

    def reformat_image(
        self, new_size: tuple[int, int], image: np.ndarray
    ) -> np.ndarray:
        if len(image.shape) == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv.cvtColor(image, cv.COLOR_BGRA2RGB)
        else:
            raise ValueError(f"Unsupported image format: shape={image.shape}")
        return cv.resize(image, new_size)

    def make_grid(self) -> np.ndarray | None:
        if not self.image_bundles:
            return None

        h, w = self._get_scaled_dimensions()
        reformatted_bundles = self._reformat_and_pad_bundles(h, w)

        if not reformatted_bundles:
            return None

        # Build grid as: rows=steps, columns=images
        columns = []
        columns.append(self._create_step_labels_column(reformatted_bundles[0], h))
        columns.extend(self._create_image_columns(reformatted_bundles, w))

        return np.hstack(columns)

    def _get_scaled_dimensions(self) -> tuple[int, int]:
        if not self.image_bundles:
            return (0, 0)
        h, w = self.image_bundles[0].source_image.shape[:2]
        return int(h * self.scale), int(w * self.scale)

    def _reformat_and_pad_bundles(self, h: int, w: int) -> list[ImageBundle]:
        reformatted_bundles = []

        for img_bundle in self.image_bundles:
            reformatted_source = self.reformat_image((w, h), img_bundle.source_image)
            reformatted_processed = {
                k: self.reformat_image((w, h), img)
                for k, img in img_bundle.processed_images.items()
            }

            num_images = 1 + len(reformatted_processed)
            padding_count = self._calculate_padding_count(num_images)

            for i in range(padding_count):
                padding_key = f"_pad_{num_images + i}"
                reformatted_processed[padding_key] = self._create_padding_image(h, w)

            reformatted_bundles.append(
                ImageBundle(
                    source_image=reformatted_source,
                    processed_images=reformatted_processed,
                    filename=img_bundle.filename,
                )
            )

        return reformatted_bundles

    def _calculate_padding_count(self, num_images: int) -> int:
        return (self.columns - (num_images % self.columns)) % self.columns

    def _create_padding_image(self, h: int, w: int) -> np.ndarray:
        return np.ones((h, w, 3), dtype=np.uint8) * PADDING_COLOR

    def _create_step_labels_column(
        self, first_bundle: ImageBundle, img_height: int
    ) -> np.ndarray:
        rows = [
            self._create_label("", LABEL_WIDTH, LABEL_HEIGHT),
            self._create_label("Original", LABEL_WIDTH, img_height),
        ]

        for description in first_bundle.processed_images.keys():
            label_text = "" if description.startswith("_pad_") else description
            rows.append(self._create_label(label_text, LABEL_WIDTH, img_height))

        return np.vstack(rows)

    def _create_image_columns(
        self, bundles: list[ImageBundle], img_width: int
    ) -> list[np.ndarray]:
        columns = []

        for bundle in bundles:
            rows = [
                self._create_label(bundle.filename, img_width, LABEL_HEIGHT),
                bundle.source_image,
            ]
            rows.extend(bundle.processed_images.values())
            columns.append(np.vstack(rows))

        return columns

    def _create_label(self, text: str, width: int, height: int) -> np.ndarray:
        label_bg = np.ones((height, width, 3), dtype=np.uint8) * PADDING_COLOR

        if text:
            self._add_centered_text(label_bg, text, width, height)

        return label_bg

    def _add_centered_text(
        self, image: np.ndarray, text: str, width: int, height: int
    ) -> None:
        font = cv.FONT_HERSHEY_SIMPLEX
        text_size = cv.getTextSize(text, font, FONT_SCALE, FONT_THICKNESS)[0]
        text_x = max(0, (width - text_size[0]) // 2)
        text_y = (height + text_size[1]) // 2
        cv.putText(
            image,
            text,
            (text_x, text_y),
            font,
            FONT_SCALE,
            (0, 0, 0),
            FONT_THICKNESS,
            cv.LINE_AA,
        )

    def show(self) -> None:
        grid = self.make_grid()
        if grid is not None:
            if self.im is None:
                self.im = self.ax.imshow(grid)
                plt.show(block=False)
            else:
                self.im.set_data(grid)
                self.im.set_extent((0, grid.shape[1], grid.shape[0], 0))
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def update(self, image_bundles: list[ImageBundle]) -> None:
        self.image_bundles = image_bundles
        self.show()
