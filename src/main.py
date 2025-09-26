from dataclasses import dataclass
import cv2
import matplotlib.pyplot as plt
import os
import glob
import importlib
import processing
import numpy as np
import traceback
from loguru import logger


@dataclass
class ImageResult:
    image: np.ndarray
    description: str

@dataclass
class DataResult:
    data: np.ndarray
    description: str

class ProcessingResult:
    def __init__(self, processed_images: list[ImageResult]=None, data: list[DataResult]=None):
        self.processed_images = processed_images if processed_images is not None else []
        self.data = data if data is not None else []

    def append_image(self, image, description: str):
        self.processed_images.append(ImageResult(image=image, description=description))

    def append_data(self, data, description: str):
        self.data.append(DataResult(data=data, description=description))


def make_grid(images: list[ImageResult], cols=3, scale=1.0):
    if not images:
        return None

    h, w = images[0].image.shape[:2]
    h, w = int(h * scale), int(w * scale)
    resized = [cv2.resize(img.image, (w, h)) for img in images]

    while len(resized) % cols != 0:
        resized.append(np.zeros_like(resized[0]))

    rows = []
    for i in range(0, len(resized), cols):
        row = np.hstack(resized[i:i+cols])
        rows.append(row)

    return np.vstack(rows)

def plot_image_results(image_results: list[ImageResult], fig_ax=None):
    plt.ion()
    grid_processed = make_grid(image_results, cols=4)
    if grid_processed is not None:
        grid_rgb = cv2.cvtColor(grid_processed, cv2.COLOR_BGR2RGB)
        if fig_ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(grid_rgb)
            ax.set_title("Processed Images Grid")
            ax.axis('off')
            plt.show(block=False)
            return fig, ax, im
        else:
            fig, ax, im = fig_ax
            im.set_data(grid_rgb)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            return fig, ax, im
    return fig_ax

def print_data_results(data_results: list[DataResult]):
    for data_result in data_results:
        logger.info(f"{data_result.description}: {data_result.data}")

def main():
    image_folder = "./images"
    image_paths = glob.glob(os.path.join(image_folder, "*.*"))

    if not image_paths:
        print("No images found in folder:", image_folder)
        return

    print(f"Loaded {len(image_paths)} images")

    processing_file = processing.__file__
    last_mtime = os.path.getmtime(processing_file)
    can_show = True

    fig_ax = None
    plt.ion()
    while True:
        # Auto-reload processing.py
        try:
            mtime = os.path.getmtime(processing_file)
            if mtime != last_mtime:
                importlib.reload(processing)
                last_mtime = mtime
                print("🔄 Auto-reloaded processing.py")
                can_show = True  # try displaying again after reload
        except Exception as e:
            print("⚠ Error reloading processing.py:", e)
            can_show = False

        processing_result = ProcessingResult(processed_images=[], data=[])

        if can_show:
            try:
                # Load and process all images
                for path in image_paths:
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    processing.process_image(processing_result, img)
                for i, processed_img in enumerate(processing_result.processed_images):
                    cv2.imwrite(f"./images/results/image_processed_{i}.png", processed_img.image)

                fig_ax = plot_image_results(processing_result.processed_images, fig_ax)
                print_data_results(processing_result.data)

            except Exception:
                print("⚠ Error in processing.py. Waiting for fix...")
                traceback.print_exc()
                can_show = False
                plt.close('all')  # hide previous images
                fig_ax = None
            can_show = False

        plt.pause(0.2)

if __name__ == "__main__":
    main()
