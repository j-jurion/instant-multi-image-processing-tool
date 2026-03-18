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
from matplotlib import pyplot as plt

from base import ImageBundle
from viewer import Viewer

# Constants
FILE_WATCH_INTERVAL = 0.1  # seconds
DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class IMIP:
    def __init__(
        self,
        images: Path,
        output_directory: Path | None = None,
        save_debug_images: bool = False,
    ):
        self.debug_images = self._load_images(images)
        self._current_image_index: int | None = None
        self._lock = threading.Lock()
        self.output_directory = output_directory
        self.save_debug_images = save_debug_images
        self._imip_reloader: IMIPReloader | None = None
        self.viewer = Viewer(self.debug_images)

    def _load_images(self, path: Path) -> list[ImageBundle]:
        if not path.is_file() and not path.is_dir():
            raise ValueError(f"Path {path} is neither a file nor a directory.")

        if path.is_file():
            image_bundle = self._load_single_image(path)
            return [image_bundle] if image_bundle else []

        # Load all images from directory
        image_bundles = []
        for img_file in sorted(path.glob("*.*")):
            if not img_file.is_file():
                continue

            # Filter by common image extensions
            if img_file.suffix.lower() not in DEFAULT_IMAGE_EXTENSIONS:
                continue

            image_bundle = self._load_single_image(img_file)
            if image_bundle:
                image_bundles.append(image_bundle)

        if not image_bundles:
            logger.warning(f"No valid images found in {path}")

        return image_bundles

    def _load_single_image(self, path: Path) -> ImageBundle | None:
        try:
            image = cv.imread(str(path))
            if image is None:
                logger.warning(
                    f"Failed to load image at {path}. File may be corrupted or unsupported format."
                )
                return None

            return ImageBundle(
                source_image=image, processed_images={}, filename=path.name
            )
        except Exception as e:
            logger.warning(f"Could not load image {path}: {e}")
            return None

    def debug_fn(self, function: Callable[[Any], Any], watch_file: Path) -> None:
        self._imip_reloader = IMIPReloader(watch_file)
        function_name = function.__name__
        watch_file_abs = watch_file.resolve()

        # Initial processing run
        self._process_all_images(function)

        # Setup file watcher
        def on_file_changed(reloaded: bool):
            if reloaded:
                logger.info(f"Detected change in {watch_file.name}, reloading...")
                reloaded_function = self._reload_function(function_name, watch_file_abs)
                if reloaded_function:
                    self._process_all_images(reloaded_function, clear_processed=True)
                else:
                    logger.error(
                        "Failed to reload function. Fix errors and save again."
                    )

        self._imip_reloader.file_reloaded_stream.subscribe(
            on_next=on_file_changed,
            on_error=lambda e: logger.error(f"File watcher error: {e}"),
        )

        # Start file watcher in background
        self._start_file_watcher()

        # Keep matplotlib on main thread
        self._show_viewer()

    def _reload_function(self, function_name: str, file_path: Path) -> Callable | None:
        logger.debug(f"Reloading function '{function_name}' from {file_path.name}")
        try:
            spec = importlib.util.spec_from_file_location(
                "__reloaded_module__", file_path
            )
            if not spec or not spec.loader:
                logger.warning(f"Could not create module spec for {file_path}")
                return None

            module = importlib.util.module_from_spec(spec)
            module.imip = self  # type: ignore - Inject IMIP instance for debugging
            spec.loader.exec_module(module)

            if not hasattr(module, function_name):
                logger.error(f"Function '{function_name}' not found in reloaded module")
                return None

            reloaded_function = getattr(module, function_name)
            logger.info(f"Successfully reloaded function '{function_name}'")
            return reloaded_function

        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path.name}: {e}")
        except Exception as e:
            logger.error(f"Failed to reload module: {e}")

        return None

    def _process_all_images(
        self, function: Callable, clear_processed: bool = False
    ) -> None:
        logger.info(f"Processing {len(self.debug_images)} image(s)")

        with self._lock:
            for i, image_bundle in enumerate(self.debug_images):
                logger.debug(
                    f"Processing image {i + 1}/{len(self.debug_images)}: {image_bundle.filename}"
                )
                self._current_image_index = i

                if clear_processed:
                    image_bundle.processed_images.clear()

                try:
                    function(image_bundle.source_image)
                except Exception as e:
                    logger.error(
                        f"Error processing image {image_bundle.filename}: {e}",
                        exc_info=True,
                    )

            self._current_image_index = None

        # Update viewer outside the lock
        self.viewer.update(self.debug_images)

        if self.save_debug_images:
            self.save_images()

    def _start_file_watcher(self) -> None:
        loop = asyncio.new_event_loop()

        def run_async_loop():
            asyncio.set_event_loop(loop)
            assert self._imip_reloader is not None
            loop.run_until_complete(self._imip_reloader.run())

        async_thread = threading.Thread(target=run_async_loop, daemon=True)
        async_thread.start()
        logger.debug("File watcher thread started")

    def _show_viewer(self) -> None:
        logger.info("Starting viewer. Close the window to exit.")
        plt.show(block=True)

    def debug(self, image: np.ndarray, description: str = "") -> None:
        assert self._current_image_index is not None, (
            "debug() must be called from within a processing function"
        )

        logger.debug(
            f"Adding processed image '{description}' for {self.debug_images[self._current_image_index].filename}"
        )
        self.debug_images[self._current_image_index].processed_images[description] = (
            image
        )

    def save_images(self) -> None:
        if not self.output_directory:
            logger.warning("Output directory not set. Cannot save images.")
            return

        os.makedirs(self.output_directory, exist_ok=True)
        saved_count = 0

        for bundle in self.debug_images:
            for desc, img in bundle.processed_images.items():
                # Create safe filename
                safe_desc = desc.replace(" ", "_").replace("/", "_")
                filename = f"{Path(bundle.filename).stem}_{safe_desc}.png"
                output_path = self.output_directory / filename

                try:
                    cv.imwrite(str(output_path), img)
                    saved_count += 1
                    logger.debug(f"Saved: {output_path}")
                except Exception as e:
                    logger.error(f"Failed to save {output_path}: {e}")

        logger.info(
            f"Saved {saved_count} processed image(s) to {self.output_directory}"
        )


class IMIPReloader:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._file_reloaded_stream: Stream[bool] = Stream()

    @property
    def file_reloaded_stream(self) -> Observable[bool]:
        return self._file_reloaded_stream

    async def run(self) -> None:
        last_mtime = None

        while True:
            try:
                current_mtime = os.path.getmtime(self.filepath)

                # Only emit when there's an actual change
                if last_mtime is not None and current_mtime != last_mtime:
                    logger.debug(f"File modification detected: {self.filepath.name}")
                    self._file_reloaded_stream.next(True)

                last_mtime = current_mtime

            except FileNotFoundError:
                logger.warning(f"Watched file not found: {self.filepath}")
            except Exception as e:
                logger.error(f"Error checking file modification time: {e}")

            await asyncio.sleep(FILE_WATCH_INTERVAL)
