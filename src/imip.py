import asyncio
import importlib
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable
import threading

import cv2 as cv
import numpy as np
from heliovision.streams.stream import Observable, Stream
from loguru import logger
from viewer import Viewer
from base import ImageBundle



class IMIP:
    def __init__(self):
        self.debug_images: list[ImageBundle] = []
        self.current_image_index: int | None = None
        self.output_directory = None
        self.save_debug_images = False
        self.imip_reloader: IMIPReloader | None = None
        self.viewer = Viewer(self.debug_images)

    def set_debug_save_dir(self, output_directory: Path | None) -> None:
        if output_directory is None:
            self.save_debug_images = False
        else:
            self.output_directory = output_directory
            self.save_debug_images = True

    def load_images(self, path: Path) -> None:
        if path.is_file():
            self._load_image(path)
        elif path.is_dir():
            for img_file in path.glob("*.*"):
                if img_file.is_file():
                    self._load_image(img_file)
        else:
            raise ValueError(f"Path {path} is neither a file nor a directory.")

    def _load_image(self, path: Path) -> None:
        try:
            image = cv.imread(str(path))
            if image is None:
                logger.warning(f"Image at {path} could not be loaded. Image is None.")
            else:
                self.debug_images.append(
                    ImageBundle(source_image=image, processed_images={})
                )
                self.viewer.update(self.debug_images)
        except Exception as e:
            logger.warning(f"Could not load image {path}: {e}")

    def debug_fn(self, function: Callable[[Any], Any], watch_file: Path) -> None:
        """Run file watcher in background thread, matplotlib stays on main thread"""
        self.imip_reloader = IMIPReloader(watch_file)
        
        # Get module and function name for reloading
        function_name = function.__name__
        watch_file_abs = watch_file.resolve()

        def run_function(reloaded: bool):
            current_function = function
            
            if reloaded:
                logger.debug("File reloaded - reloading module")
                # Reload the module to get the updated function
                try:
                    # Create a module spec from the file path
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("__reloaded_module__", watch_file_abs)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        # Inject the imip instance into the module's global namespace
                        module.imip = self  # type: ignore
                        spec.loader.exec_module(module)
                        # Get the updated function from the reloaded module
                        current_function = getattr(module, function_name)
                        logger.debug(f"Successfully reloaded function {function_name}")
                    else:
                        logger.warning(f"Could not create spec for {watch_file_abs}")
                except Exception as e:
                    logger.error(f"Failed to reload module: {e}")
            
            # Process images (both initial run and on reload)
            logger.debug("Processing all images" + (" with updated function" if reloaded else ""))
            for i, image_bundle in enumerate(self.debug_images):
                logger.debug(f"Processing image {i + 1}/{len(self.debug_images)}")
                self.current_image_index = i
                # Clear previous processed images for this bundle on reload
                if reloaded:
                    image_bundle.processed_images.clear()
                logger.debug(image_bundle)
                current_function(image_bundle.source_image)
            self.viewer.update(self.debug_images)
        
        run_function(False)  # Initial run (don't reload on first execution)

        self.imip_reloader.file_reloaded_stream.subscribe(
            on_next=run_function,
            on_error=lambda e: logger.error(e),
        )

        # Run asyncio event loop in background thread
        loop = asyncio.new_event_loop()
        
        def run_async_loop():
            asyncio.set_event_loop(loop)
            assert self.imip_reloader is not None
            loop.run_until_complete(self.imip_reloader.run())
        
        async_thread = threading.Thread(target=run_async_loop, daemon=True)
        async_thread.start()
        
        # Keep matplotlib on main thread
        from matplotlib import pyplot as plt
        plt.show(block=True)

    def debugger(self, image: np.ndarray, description: str = ""):
        assert self.current_image_index is not None, "No current image data set."
        logger.error(f"Debugging image at index {self.current_image_index} with description '{description}'")
        self.debug_images[self.current_image_index].processed_images[description] = (
            image
        )
        self.viewer.update(self.debug_images)




class IMIPReloader:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._file_reloaded_stream: Stream[bool] = Stream()

    @property
    def file_reloaded_stream(self) -> Observable[bool]:
        return self._file_reloaded_stream

    async def run(self):
        last_mtime = None
        while True:
            current_mtime = os.path.getmtime(self.filepath)
            # Only emit when there's an actual change (not on first check or when unchanged)
            if last_mtime is not None and current_mtime != last_mtime:
                self._file_reloaded_stream.next(True)
            last_mtime = current_mtime
            await asyncio.sleep(0.1)

