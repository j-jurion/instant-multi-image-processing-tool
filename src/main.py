import glob
import importlib
import os
import traceback

import click
import cv2
import matplotlib.pyplot as plt

import processing
from base import ImageData, ProcessedImageData
from viewer import Viewer


def get_processed_images(original_images: list[ImageData]) -> list[ProcessedImageData]:
    processing_results = []
    for original_image in original_images:
        processed_image = processing.process_image(original_image.image)
        processing_results.append(
            ProcessedImageData(
                id=original_image.id,
                image=processed_image,
                original_image_id=original_image.id,
                algorithm="Example Algorithm",
            )
        )

    return processing_results


@click.command()
@click.option(
    "--image-folder", default="./images", help="Folder containing images to process"
)
@click.option(
    "--processing-file",
    default="./processing.py",
    help="Image processing Python file directory",
)
def main(image_folder, processing_file):
    image_paths = glob.glob(os.path.join(image_folder, "*.*"))

    original_images = []
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is not None:
            original_images.append(ImageData(id=i, image=img))

    viewer = Viewer(original_images=original_images)

    processing_file = processing.__file__
    last_mtime = os.path.getmtime(processing_file)

    viewer.update(processed_images=get_processed_images(original_images))

    while True:
        # Auto-reload processing.py
        try:
            mtime = os.path.getmtime(processing_file)
            if mtime != last_mtime:
                importlib.reload(processing)
                last_mtime = mtime
                print("🔄 Auto-reloaded processing.py")
                try:
                    viewer.update(
                        processed_images=get_processed_images(original_images)
                    )
                except Exception as e:
                    print("⚠ Error displaying images:", e)
                    traceback.print_exc()
        except Exception as e:
            print("⚠ Error reloading processing.py:", e)

        if not plt.fignum_exists(viewer.fig.number):
            print("Window closed. Exiting.")
            break
        plt.pause(0.2)


if __name__ == "__main__":
    main()
