import pdb
import sys

import cv2 as cv
import numpy as np


def show_images(*args, **kwargs):
    frame = sys._getframe(1)
    locals_dict = frame.f_locals

    for name, value in locals_dict.items():
        if isinstance(value, np.ndarray) and len(value.shape) >= 2:
            cv.imshow(f"{name}", value)

    cv.waitKey(0)
    cv.destroyAllWindows()


def _show_images_in_frame(frame, specific_var=None):
    """
    Display all numpy arrays in the given frame

    Args:
        frame: The frame to inspect
        specific_var: If provided, only show images from this specific variable
                     (handles lists/tuples of images)
    """
    locals_dict = frame.f_locals
    images_found = []

    if specific_var and specific_var in locals_dict:
        # Show images from specific variable
        value = locals_dict[specific_var]

        # If it's a list/tuple of images, show each one
        if isinstance(value, (list, tuple)):
            for i, img in enumerate(value):
                if isinstance(img, np.ndarray) and len(img.shape) >= 2:
                    name = f"{specific_var}[{i}]"
                    images_found.append(name)
                    cv.imshow(name, img)
                    cv.waitKey(1)
        # If it's a single image, show it
        elif isinstance(value, np.ndarray) and len(value.shape) >= 2:
            images_found.append(specific_var)
            cv.imshow(specific_var, value)
            cv.waitKey(1)
    else:
        # Show all numpy arrays in frame
        for name, value in locals_dict.items():
            if isinstance(value, np.ndarray) and len(value.shape) >= 2:
                images_found.append(name)
                cv.imshow(f"{name}", value)
                cv.waitKey(1)  # Allow window to render

    if images_found:
        print(f"\n[Visualizer] Displaying: {', '.join(images_found)}")
        print("[Visualizer] Press any key to close all windows...")
        cv.waitKey(0)
        cv.destroyAllWindows()


def debug_with_images(*args, **kwargs):
    """Shows images then enters pdb debugger"""
    show_images()
    import pdb

    pdb.set_trace()


class ImageDebugger(pdb.Pdb):
    """Custom debugger that automatically shows images at each breakpoint"""

    def user_line(self, frame):
        """Called when debugger stops at a line - auto-show images"""
        _show_images_in_frame(frame)
        super().user_line(frame)


def auto_visualize_breakpoint(*args, **kwargs):
    """Custom breakpoint that uses ImageDebugger only when in debug mode"""
    # Check if we're running under a debugger
    in_debugger = sys.gettrace() is not None or "debugpy" in sys.modules

    if in_debugger:
        # Use custom debugger with auto-visualization
        debugger = ImageDebugger()
        debugger.set_trace(sys._getframe(1))
    else:
        # Do nothing when running freely
        pass


# Variable-watching trace function
_original_trace = None
_watched_variable = "imip_images"
_last_seen_value = {}  # Track what we've seen per frame


def _watch_variable_trace(frame, event, arg):
    """Trace function that watches for imip_images variable creation/update"""
    global _last_seen_value

    if event == "line":
        frame_id = id(frame)
        locals_dict = frame.f_locals

        # Check if the watched variable exists in this frame
        if _watched_variable in locals_dict:
            current_value = id(locals_dict[_watched_variable])

            # If we haven't seen this value in this frame before, show images
            if (
                frame_id not in _last_seen_value
                or _last_seen_value.get(frame_id) != current_value
            ):
                _last_seen_value[frame_id] = current_value
                print(f"\n[Visualizer] Detected '{_watched_variable}' variable!")
                _show_images_in_frame(frame, specific_var=_watched_variable)
        elif frame_id in _last_seen_value:
            # Variable no longer exists, clean up tracking
            del _last_seen_value[frame_id]

    # Call original trace function if it exists
    if _original_trace:
        return _original_trace(frame, event, arg)

    return _watch_variable_trace


def enable_auto_visualization_on_variable(variable_name="imip_images"):
    """
    Enable automatic image visualization when a specific variable is created/updated.
    Only works in debug mode (when debugpy is active).

    Args:
        variable_name: The variable name to watch for (default: "imip_images")
    """
    global _original_trace, _watched_variable, _last_seen_value

    _watched_variable = variable_name
    _last_seen_value = {}

    # Only enable if debugpy is loaded (VS Code debugger is active)
    if "debugpy" in sys.modules:
        _original_trace = sys.gettrace()
        sys.settrace(_watch_variable_trace)
        print("=" * 60)
        print("[Visualizer] Auto-visualization ENABLED!")
        print("=" * 60)
        print(f"Watching for variable: '{variable_name}'")
        print(
            f"When '{variable_name}' is created/updated, images will show automatically"
        )
        print("=" * 60)
    else:
        print("[Visualizer] Not in debug mode - auto-visualization disabled")


def disable_auto_visualization():
    """Disable automatic image visualization"""
    global _original_trace, _last_seen_value
    if _original_trace:
        sys.settrace(_original_trace)
        _original_trace = None
        _last_seen_value = {}
        print("[Visualizer] Auto-visualization disabled")


# Simpler approach for VS Code integration
def show_and_pause():
    """
    Use this in VS Code conditional breakpoints.

    Right-click red dot → Edit Breakpoint → Expression (not condition)
    Enter: show_and_pause()

    Or just call from debug console when paused at any red dot.
    """
    import sys

    frame = sys._getframe(1)
    _show_images_in_frame(frame)
    return False  # Return False so breakpoint still triggers


def enable_vscode_breakpoint_visualization():
    """Deprecated - use enable_auto_visualization_on_variable() instead"""
    enable_auto_visualization_on_variable("imip_images")


def disable_vscode_breakpoint_visualization():
    """Deprecated - use disable_auto_visualization() instead"""
    disable_auto_visualization()


def show_current_images():
    """Manually show images from current frame - call from debug console"""
    frame = sys._getframe(1)
    _show_images_in_frame(frame)


# Short alias for convenience
show = show_current_images  # Type just: show()
