import asyncio
import importlib.util
import os
import threading
from pathlib import Path
from typing import Any, Callable

import cv2 as cv
import numpy as np
from heliovision.streams.stream import Observable, Stream
from loguru import logger

from base import ImageBundle
from viewer import Viewer


class IMIP:
    def __init__(self, images: Path, output_directory: Path | None = None, save_debug_images: bool = False):
        self.debug_images = self.load_images(images)
        self.current_image_index: int | None = None
        self.output_directory = output_directory
        self.save_debug_images = save_debug_images
        self.imip_reloader: IMIPReloader | None = None
        self.viewer = Viewer(self.debug_images)

    def load_images(self, path: Path) -> list[ImageBundle]:
        """Load images from a file or directory."""
        if not path.is_file() and not path.is_dir():
            raise ValueError(f"Path {path} is neither a file nor a directory.")
        
        if path.is_file():
            image_bundle = self._load_image(path)
            return [image_bundle] if image_bundle else []
        
        image_bundles = []
        for img_file in sorted(path.glob("*.*")):
            if not img_file.is_file():
                continue
            image_bundle = self._load_image(img_file)
            if image_bundle:
                image_bundles.append(image_bundle)
        return image_bundles

    def _load_image(self, path: Path) -> ImageBundle | None:
        """Load a single image file."""
        try:
            image = cv.imread(str(path))
            if image is None:
                logger.warning(f"Image at {path} could not be loaded. Image is None.")
            else:
                return ImageBundle(
                        source_image=image,
                        processed_images={},
                        filename=path.name
                    )
        except Exception as e:
            logger.warning(f"Could not load image {path}: {e}")

    def debug_fn(self, function: Callable[[Any], Any], watch_file: Path) -> None:
        """Watch a file for changes and rerun the processing function on all images."""
        self.imip_reloader = IMIPReloader(watch_file)
        function_name = function.__name__
        watch_file_abs = watch_file.resolve()

        # Initial processing run
        self._process_all_images(function)

        # Setup file watcher
        def on_file_changed(reloaded: bool):
            if reloaded:
                reloaded_function = self._reload_function(function_name, watch_file_abs)
                if reloaded_function:
                    self._process_all_images(reloaded_function, clear_processed=True)

        self.imip_reloader.file_reloaded_stream.subscribe(
            on_next=on_file_changed,
            on_error=lambda e: logger.error(f"File watcher error: {e}"),
        )

        # Start file watcher in background
        self._start_file_watcher()
        
        # Keep matplotlib on main thread
        self._show_viewer()

    def _reload_function(self, function_name: str, file_path: Path) -> Callable | None:
        """Reload a function from a file."""
        logger.debug("File reloaded - reloading module")
        try:
            spec = importlib.util.spec_from_file_location("__reloaded_module__", file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                module.imip = self  # type: ignore
                spec.loader.exec_module(module)
                reloaded_function = getattr(module, function_name)
                logger.debug(f"Successfully reloaded function {function_name}")
                return reloaded_function
            else:
                logger.warning(f"Could not create spec for {file_path}")
        except Exception as e:
            logger.error(f"Failed to reload module: {e}")
        return None

    def _process_all_images(self, function: Callable, clear_processed: bool = False) -> None:
        """Process all loaded images with the given function."""
        logger.debug(f"Processing {len(self.debug_images)} images")
        for i, image_bundle in enumerate(self.debug_images):
            logger.debug(f"Processing image {i + 1}/{len(self.debug_images)}")
            self.current_image_index = i
            
            if clear_processed:
                image_bundle.processed_images.clear()
            
            try:
                function(image_bundle.source_image)
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
        
        self.viewer.update(self.debug_images)

    def _start_file_watcher(self) -> None:
        """Start the file watcher in a background thread."""
        loop = asyncio.new_event_loop()
        
        def run_async_loop():
            asyncio.set_event_loop(loop)
            assert self.imip_reloader is not None
            loop.run_until_complete(self.imip_reloader.run())
        
        async_thread = threading.Thread(target=run_async_loop, daemon=True)
        async_thread.start()

    def _show_viewer(self) -> None:
        """Show the viewer window (blocking)."""
        from matplotlib import pyplot as plt
        plt.show(block=True)

    def debugger(self, image: np.ndarray, description: str = "") -> None:
        """Store a processed image for debugging and update the viewer."""
        assert self.current_image_index is not None, "No current image data set."
        logger.debug(f"Debugging image at index {self.current_image_index} with description '{description}'")
        self.debug_images[self.current_image_index].processed_images[description] = image


class IMIPReloader:
    """Watches a file for modifications and emits events when changes are detected."""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._file_reloaded_stream: Stream[bool] = Stream()

    @property
    def file_reloaded_stream(self) -> Observable[bool]:
        return self._file_reloaded_stream

    async def run(self) -> None:
        """Poll the file for changes and emit events when modified."""
        last_mtime = None
        while True:
            try:
                current_mtime = os.path.getmtime(self.filepath)
                # Only emit when there's an actual change
                if last_mtime is not None and current_mtime != last_mtime:
                    logger.debug(f"File {self.filepath.name} changed")
                    self._file_reloaded_stream.next(True)
                last_mtime = current_mtime
            except Exception as e:
                logger.error(f"Error checking file modification time: {e}")
            
            await asyncio.sleep(0.1)

