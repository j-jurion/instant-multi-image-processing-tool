import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from base import ImageBundle


class Viewer:
    def __init__(self, image_bundles: list[ImageBundle]):
        self.image_bundles = image_bundles
        self.scale = 1.0
        self.columns = 4
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.axis("off")
        self.im = None

    def get_columns(self, image_bundles: list[ImageBundle]) -> int:
        max_images = 0
        for bundle in image_bundles:
            num_images = 1 + len(bundle.processed_images)  # source + processed
            if num_images > max_images:
                max_images = num_images
        return max_images

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
        if not self.image_bundles:
            return None

        h, w = self._get_scaled_dimensions()
        reformatted_bundles = self._reformat_and_pad_bundles(h, w)

        # Build grid as: rows=steps, columns=images
        columns = []
        columns.append(self._create_step_labels_column(reformatted_bundles[0], h))
        columns.extend(self._create_image_columns(reformatted_bundles, w, h))

        return np.hstack(columns)

    def _get_scaled_dimensions(self) -> tuple[int, int]:
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
            reformatted_bundles.append(
                ImageBundle(
                    source_image=reformatted_source,
                    processed_images=reformatted_processed,
                    filename=img_bundle.filename,
                )
            )

        # Pad each row to match column count
        for bundle in reformatted_bundles:
            num_images = 1 + len(bundle.processed_images)
            padding_count = (self.columns - (num_images % self.columns)) % self.columns
            for i in range(padding_count):
                padding_key = f"_pad_{num_images + i}"
                bundle.processed_images[padding_key] = (
                    np.ones((h, w, 3), dtype=np.uint8) * 255
                )

        return reformatted_bundles

    def _create_step_labels_column(
        self, first_bundle: ImageBundle, img_height: int
    ) -> np.ndarray:
        label_width = 150
        label_height = 40

        rows = [
            self._create_text_only_label(
                "", label_width, label_height
            ),  # Top-left corner
            self._create_text_only_label("Original", label_width, img_height),
        ]

        for description in first_bundle.processed_images.keys():
            label_text = "" if description.startswith("_pad_") else description
            rows.append(
                self._create_text_only_label(label_text, label_width, img_height)
            )

        return np.vstack(rows)

    def _create_image_columns(
        self, bundles: list[ImageBundle], img_width: int, img_height: int
    ) -> list[np.ndarray]:
        label_height = 40
        columns = []

        for bundle in bundles:
            rows = [
                self._create_text_only_label(bundle.filename, img_width, label_height),
                bundle.source_image,
            ]
            rows.extend(bundle.processed_images.values())
            columns.append(np.vstack(rows))

        return columns

    def _create_text_only_label(self, text: str, width: int, height: int) -> np.ndarray:
        label_bg = np.ones((height, width, 3), dtype=np.uint8) * 255

        if text:
            self._add_centered_text(label_bg, text, width, height)

        return label_bg

    def _create_filename_label(self, text: str, width: int, height: int) -> np.ndarray:
        label_bg = np.ones((height, width, 3), dtype=np.uint8) * 255

        if text:
            self._add_centered_text(label_bg, text, width, height)

        return label_bg

    def _add_centered_text(
        self, image: np.ndarray, text: str, width: int, height: int
    ) -> None:
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv.putText(
            image,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv.LINE_AA,
        )

    def show(self):
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

    def update(self, image_bundles: list[ImageBundle]):
        self.image_bundles = image_bundles
        self.show()
