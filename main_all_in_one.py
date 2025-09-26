import cv2
import os
import glob
import importlib
import img_processing.processing as processing
import numpy as np
import traceback

def make_grid(images, cols=3, scale=1.0):
    if not images:
        return None

    h, w = images[0].shape[:2]
    h, w = int(h * scale), int(w * scale)
    resized = [cv2.resize(img, (w, h)) for img in images]

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

        originals = []
        processed = []

        if can_show:
            try:
                # Load and process all images
                for path in image_paths:
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    originals.append(img)
                    processed.append(processing.process_image(img))
                for i, processed_img in enumerate(processed):
                    cv2.imwrite(f"./images/results/image_processed_{i}.png", processed_img)

                # grid_original = make_grid(originals, cols=4)
                grid_processed = make_grid(processed, cols=4)

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
