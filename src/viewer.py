import time
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
        plt.axis("off")
        self.im = None

    def get_columns(self, image_bundles) -> int:
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
        
        rows = []
        rows.append(self._create_header_row(reformatted_bundles[0], w))
        rows.extend(self._create_image_rows(reformatted_bundles, h))
        
        return np.vstack(rows)
    
    def _get_scaled_dimensions(self) -> tuple[int, int]:
        """Get the scaled height and width for images."""
        h, w = self.image_bundles[0].source_image.shape[:2]
        return int(h * self.scale), int(w * self.scale)
    
    def _reformat_and_pad_bundles(self, h: int, w: int) -> list[ImageBundle]:
        """Reformat all images to the same size and add padding to match column count."""
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
                    filename=img_bundle.filename
                )
            )
        
        # Pad each row to match column count
        for bundle in reformatted_bundles:
            num_images = 1 + len(bundle.processed_images)
            padding_count = (self.columns - (num_images % self.columns)) % self.columns
            for i in range(padding_count):
                padding_key = f'_pad_{num_images + i}'
                bundle.processed_images[padding_key] = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        return reformatted_bundles
    
    def _create_header_row(self, first_bundle: ImageBundle, img_width: int) -> np.ndarray:
        """Create the header row with column labels."""
        text_height = 40
        filename_width = 150
        
        header_parts = [
            np.ones((text_height, filename_width, 3), dtype=np.uint8) * 255,
            self._create_text_only_label("Original Image", img_width, text_height)
        ]
        
        for description in first_bundle.processed_images.keys():
            label_text = "" if description.startswith('_pad_') else description
            header_parts.append(self._create_text_only_label(label_text, img_width, text_height))
        
        return np.hstack(header_parts)
    
    def _create_image_rows(self, bundles: list[ImageBundle], img_height: int) -> list[np.ndarray]:
        """Create all image rows with filename labels."""
        filename_width = 150
        rows = []
        
        for bundle in bundles:
            row_parts = [
                self._create_filename_label(bundle.filename, filename_width, img_height),
                bundle.source_image
            ]
            row_parts.extend(bundle.processed_images.values())
            rows.append(np.hstack(row_parts))
        
        return rows
    
    def _create_text_only_label(self, text: str, width: int, height: int) -> np.ndarray:
        """Create a text label without an image underneath."""
        label_bg = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        if text:
            self._add_centered_text(label_bg, text, width, height)
        
        return label_bg
    
    def _create_filename_label(self, text: str, width: int, height: int) -> np.ndarray:
        """Create a vertical label with filename text."""
        label_bg = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        if text:
            self._add_centered_text(label_bg, text, width, height)
        
        return label_bg
    
    def _add_centered_text(self, image: np.ndarray, text: str, width: int, height: int) -> None:
        """Add centered text to an image in-place."""
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv.LINE_AA)

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


if __name__ == "__main__":
    img1 = cv.imread("images/01.jpg")
    img2 = cv.imread("images/02.jpg")

    assert img1 is not None
    assert img2 is not None

    bundle1 = ImageBundle(source_image=img1, processed_images={"1": img1})
    bundle2 = ImageBundle(source_image=img2, processed_images={"1": img2, "2": img2})

    viewer = Viewer([bundle1, bundle2])
    viewer.show()
    time.sleep(20)

    bundle1 = ImageBundle(source_image=img1, processed_images={"1": img1, "2": img1})

    viewer.update([bundle1, bundle2])
    
    plt.show(block=True)  # Block to keep window open
