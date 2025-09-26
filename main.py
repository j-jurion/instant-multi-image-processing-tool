import cv2
import os
import glob
import importlib
import img_processing.processing_sobel as processing_sobel  # your custom processing module
import time

def main():
    image_folder = "../../images"  # folder with input images
    image_paths = glob.glob(os.path.join(image_folder, "*.*"))

    if not image_paths:
        print("No images found in folder:", image_folder)
        return

    print(f"Loaded {len(image_paths)} images")

    # Track file modification time for auto-reload
    processing_file = processing_sobel.__file__
    last_mtime = os.path.getmtime(processing_file)

    idx = 0
    while True:
        # Auto-reload if processing.py has changed
        try:
            mtime = os.path.getmtime(processing_file)
            if mtime != last_mtime:
                importlib.reload(processing_sobel)
                last_mtime = mtime
                print("🔄 Auto-reloaded processing.py")
        except FileNotFoundError:
            pass  # if file temporarily missing during save

        # Load current image
        img = cv2.imread(image_paths[idx])
        if img is None:
            print(f"Failed to load {image_paths[idx]}")
            idx = (idx + 1) % len(image_paths)
            continue

        # Process with the latest version of processing.py
        processed = processing_sobel.process_image(img)

        # Show original + processed
        cv2.imshow("Original", img)
        cv2.imshow("Processed", processed)

        key = cv2.waitKey(100) & 0xFF

        if key == ord("q"):  # quit
            break
        elif key == ord("n"):  # next image
            idx = (idx + 1) % len(image_paths)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
