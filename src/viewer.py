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
        if len(self.image_bundles) == 0:
            return None
        h, w = self.image_bundles[0].source_image.shape[:2]
        h, w = int(h * self.scale), int(w * self.scale)

        reformatted_bundles = []    
        for img_bundle in self.image_bundles:
            reformatted_source = self.reformat_image((w, h), img_bundle.source_image)
            reformatted_processed = {}
            for k, img in img_bundle.processed_images.items():
                reformatted_processed[k] = self.reformat_image((w, h), img)
            reformatted_bundles.append(
                ImageBundle(
                    source_image=reformatted_source,
                    processed_images=reformatted_processed,
                )   
            )

        # Pad each row to match column count
        for bundle in reformatted_bundles:
            num_images = 1 + len(bundle.processed_images)  # source + processed
            while num_images % self.columns != 0:
                padding_key = f'_pad_{num_images}'
                bundle.processed_images[padding_key] = np.ones((h, w, 3), dtype=np.uint8) * 255
                num_images += 1

        rows = []
        for bundle in reformatted_bundles:
            row = [bundle.source_image]
            for img in bundle.processed_images.values():
                row.append(img)
            
            rows.append(np.hstack(row))

        return np.vstack(rows)

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
