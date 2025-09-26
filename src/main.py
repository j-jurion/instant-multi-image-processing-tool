from dataclasses import dataclass
import cv2
import os
import glob
import importlib
import processing
import numpy as np
import traceback


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

                # grid_original = make_grid(originals, cols=4)
                grid_processed = make_grid(processing_result.processed_images, cols=4)

                # if grid_original is not None and grid_processed is not None:
                #     combined = np.hstack([grid_original, grid_processed])
                #     cv2.imshow("Original (left) | Processed (right)", combined)
                if grid_processed is not None:
                    combined = np.hstack([grid_processed])
                    cv2.imshow("Processed", combined)
                    

            
            except Exception:
                print("⚠ Error in processing.py. Waiting for fix...")
                traceback.print_exc()
                can_show = False
                cv2.destroyAllWindows()  # hide previous images
            can_show = False

        key = cv2.waitKey(200) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
