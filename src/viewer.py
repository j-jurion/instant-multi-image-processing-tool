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
DEFAULT_MAX_PAGE_WIDTH = (
    2500  # Maximum total width in pixels for all images on one page
)
PADDING_COLOR = 255


class Viewer:
    def __init__(
        self,
        image_bundles: list[ImageBundle],
        max_page_width: int = DEFAULT_MAX_PAGE_WIDTH,
    ):
        self.image_bundles = image_bundles
        self.scale = DEFAULT_SCALE
        self.columns = DEFAULT_COLUMNS
        self.max_page_width = max_page_width
        self.current_offset = 0
        self._setup_matplotlib()

    def _setup_matplotlib(self) -> None:
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.axis("off")
        self.im = None
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _on_key_press(self, event) -> None:
        """Handle keyboard events for navigation."""
        if event.key == "right":
            self._next_page()
        elif event.key == "left":
            self._previous_page()
        elif event.key == "escape":
            plt.close(self.fig)

    def _calculate_visible_count(self) -> int:
        """Calculate how many image bundles can fit within the max page width."""
        if not self.image_bundles:
            return 0

        h, w = self._get_scaled_dimensions()
        available_width = self.max_page_width - LABEL_WIDTH

        if w <= 0:
            return 1

        # Calculate how many image columns fit in the available width
        max_bundles = max(1, available_width // w)
        return min(max_bundles, len(self.image_bundles) - self.current_offset)

    def _next_page(self) -> None:
        """Navigate to the next page of images."""
        if self.current_offset >= len(self.image_bundles):
            return

        visible_count = self._calculate_visible_count()
        new_offset = self.current_offset + visible_count

        if new_offset < len(self.image_bundles):
            self.current_offset = new_offset
            self.show()

    def _previous_page(self) -> None:
        """Navigate to the previous page of images."""
        if self.current_offset <= 0:
            return

        # Calculate how many images would have been on the previous page
        h, w = self._get_scaled_dimensions()
        available_width = self.max_page_width - LABEL_WIDTH
        bundles_per_page = max(1, available_width // w) if w > 0 else 1

        self.current_offset = max(0, self.current_offset - bundles_per_page)
        self.show()

    def _get_visible_bundles(self) -> list[ImageBundle]:
        """Get the currently visible slice of image bundles."""
        visible_count = self._calculate_visible_count()
        end_idx = min(self.current_offset + visible_count, len(self.image_bundles))
        return self.image_bundles[self.current_offset : end_idx]

    def _get_page_info(self) -> str:
        """Get current page information string."""
        if not self.image_bundles:
            return "0 of 0"
        start = self.current_offset + 1
        visible_count = self._calculate_visible_count()
        end = min(self.current_offset + visible_count, len(self.image_bundles))
        total = len(self.image_bundles)
        return f"{start}-{end} of {total}"

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

        visible_bundles = self._get_visible_bundles()
        if not visible_bundles:
            return None

        h, w = self._get_scaled_dimensions()
        reformatted_bundles = self._reformat_and_pad_bundles(h, w, visible_bundles)

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

    def _reformat_and_pad_bundles(
        self, h: int, w: int, bundles: list[ImageBundle]
    ) -> list[ImageBundle]:
        reformatted_bundles = []

        for img_bundle in bundles:
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

            # Update window title with page info
            page_info = self._get_page_info()
            if self.fig.canvas.manager is not None:
                self.fig.canvas.manager.set_window_title(
                    f"Images: {page_info} (Arrow keys: navigate | ESC: close)"
                )

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def update(self, image_bundles: list[ImageBundle]) -> None:
        self.image_bundles = image_bundles
        # Keep current page if valid, otherwise adjust to last valid page
        if self.current_offset >= len(self.image_bundles):
            self.current_offset = max(0, len(self.image_bundles) - 1)
        self.show()
