from dataclasses import dataclass, field
import os
from typing import Any
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class DebugImage:
    image: np.ndarray
    description: str = ""


@dataclass
class ImageData:
    index: int
    image: np.ndarray
    debug_images: list[DebugImage] = field(default_factory=list)
    result_image: np.ndarray | None = None
    result_data: float | None = None


def show_image(image: np.ndarray, title: str = "Image"):
    """Display an image using matplotlib."""
    plt.figure()
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert BGR to RGB for matplotlib
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
    else:
        # Grayscale image
        plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_multiple_images(data_list: list[ImageData], cols: int = 3):
    """Display multiple images in a non-blocking window using matplotlib subplots.
    Each ImageData gets its own row with its debug images.
    
    Args:
        data_list: List of ImageData objects
        cols: Number of columns per row
    """
    plt.ion()
    
    # Calculate total rows needed (one row per ImageData)
    total_rows = len(data_list)
    if total_rows == 0:
        return
    
    # Find max number of debug images to determine figure height
    max_debug_images = max(len(data.debug_images) for data in data_list) if data_list else 0
    actual_cols = min(cols, max_debug_images) if max_debug_images > 0 else 1
    
    fig, axes = plt.subplots(total_rows, actual_cols, figsize=(actual_cols * 4, total_rows * 4))
    
    # Ensure axes is always 2D array
    if total_rows == 1 and actual_cols == 1:
        axes = np.array([[axes]])
    elif total_rows == 1:
        axes = axes.reshape(1, -1)
    elif actual_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for row_idx, img_data in enumerate(data_list):
        debug_images = img_data.debug_images
        
        for col_idx in range(actual_cols):
            ax = axes[row_idx, col_idx]
            
            if col_idx < len(debug_images):
                debug_image = debug_images[col_idx]
                image = debug_image.image
                
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Convert BGR to RGB for matplotlib
                    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    ax.imshow(image_rgb)
                else:
                    # Grayscale image
                    ax.imshow(image, cmap='gray')
                
                ax.set_title(f"Image {img_data.index} - {col_idx}: {debug_image.description}")
            else:
                # Hide empty subplot
                ax.axis('off')
            
            ax.axis('off')
    
    plt.tight_layout()
    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()



class IMIP:
    def __init__(self):
        self.data: list[ImageData] = []
        self.current_data_index: int | None = None
        self.output_directory: str = ""
        self.debug_images_saved: bool = False
        
    def load(self, directory: str):
        self.clear_data()
        print(f"Loading images from: {directory}")
        for file in os.listdir(directory):
            if file.endswith('.bmp') or file.endswith('.png') or file.endswith('.jpg'):
                filepath = os.path.join(directory, file)
                image = cv.imread(filepath)
                if image is not None:
                    index = len(self.data)
                    self.data.append(ImageData(index=index, image=image))

    def loaded_images(self) -> list[ImageData]:
        return self.data
    
    def clear_data(self):
        self.data.clear()
        self.current_data_index = None
    
    def register_result(self, result_image: np.ndarray, result_data: dict) :
        assert self.current_data_index is not None, "No current image data set."
        self.data[self.current_data_index].result_data = result_data
        self.data[self.current_data_index].result_image = result_image
    
    #
    # Debugger functions -----------------------------------------------------------
    #

    def debugger(self, image: np.ndarray, description: str = ""):
        assert self.current_data_index is not None, "No current image data set."
        self.data[self.current_data_index].debug_images.append(DebugImage(image=image, description=description))
        if self.debug_images_saved:
            self.save_debug_images(self.output_directory)

    def show_debug_images(self, block=True):
        assert self.current_data_index is not None, "No current image data set."
        plt.close('all')
        show_multiple_images(self.data)

    def set_debugger_options(self, output_directory: str, debug_images_saved: bool):
        self.output_directory = output_directory
        self.debug_images_saved = debug_images_saved

    def save_debug_images(self, output_directory: str):
        assert self.current_data_index is not None, "No current image data set."
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        for img_data in self.data:
            for i, debug_image in enumerate(img_data.debug_images):
                output_path = os.path.join(output_directory, f"Image_{img_data.index}_Debug_{i}.png")
                cv.imwrite(output_path, debug_image.image)

    #
    # Instant processing functions -----------------------------------------------------------
    #

    def debug_fn(self, function, watch_file: str):
        """
        Run function on all images. If watch_file is provided, monitor it for changes and rerun.
        """
        
        def run_function():
            for i, image_data in enumerate(self.loaded_images()):
                image_data.debug_images.clear()
                self.current_data_index = i
                function(image_data.image)
            self.show_debug_images(block=False)
        
        while True:
            if self.file_is_reloaded(watch_file):
                print("🔄 File changed, reloading...")
                run_function()
            plt.pause(1)


    def file_is_reloaded(self, filepath: str) -> bool:
        """Check if a file has been modified since last check."""
        current_mtime = os.path.getmtime(filepath)
        if not hasattr(self, '_file_mtimes'):
            self._file_mtimes = {}
        last_mtime = self._file_mtimes.get(filepath, None)
        self._file_mtimes[filepath] = current_mtime
        return last_mtime is None or current_mtime != last_mtime
    

    #
    # Test functions -----------------------------------------------------------
    #

    def test_fn(self, function)-> list[Any]:
        """
        Run function on all images for testing purposes.
        """
        results = []
        for i, image_data in enumerate(self.loaded_images()):
            image_data.debug_images.clear()
            self.current_data_index = i
            # self.register_result(*function(image_data.image))
            results.append(function(image_data.image))
        return results

    def results(self) -> list[float | None]:
        return [data.result_data for data in self.data]

imip = IMIP()

