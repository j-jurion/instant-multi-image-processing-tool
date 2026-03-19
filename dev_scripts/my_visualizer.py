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
_accumulated_images = []  # Store images across multiple calls


def _watch_variable_trace(frame, event, arg):
    """Trace function that watches for imip_images variable creation/update"""
    global _last_seen_value, _accumulated_images

    if event == "line":
        frame_id = id(frame)
        locals_dict = frame.f_locals

        # Check if the watched variable exists in this frame
        if _watched_variable in locals_dict:
            current_value = id(locals_dict[_watched_variable])

            # If we haven't seen this value in this frame before, accumulate images
            if (
                frame_id not in _last_seen_value
                or _last_seen_value.get(frame_id) != current_value
            ):
                _last_seen_value[frame_id] = current_value
                value = locals_dict[_watched_variable]

                # Accumulate images (don't display)
                print(
                    f"\n[Visualizer] Detected '{_watched_variable}' - accumulating..."
                )
                if isinstance(value, (list, tuple)):
                    for img in value:
                        if isinstance(img, np.ndarray) and len(img.shape) >= 2:
                            _accumulated_images.append(img.copy())
                elif isinstance(value, np.ndarray) and len(value.shape) >= 2:
                    _accumulated_images.append(value.copy())
                print(
                    f"[Visualizer] Total accumulated: {len(_accumulated_images)} images"
                )
        elif frame_id in _last_seen_value:
            # Variable no longer exists, clean up tracking
            del _last_seen_value[frame_id]

    # Call original trace function if it exists
    if _original_trace:
        return _original_trace(frame, event, arg)

    return _watch_variable_trace


def enable_auto_visualization_on_variable(variable_name="imip_images"):
    """
    Enable automatic image accumulation when a specific variable is created/updated.
    Only works in debug mode (when debugpy is active).

    Args:
        variable_name: The variable name to watch for (default: "imip_images")

    Images are accumulated automatically but NOT displayed.
    Call show_accumulated() to display all images - this is the ONLY way to see them.
    """
    global _original_trace, _watched_variable, _last_seen_value

    _watched_variable = variable_name
    _last_seen_value = {}

    # Only enable if debugpy is loaded (VS Code debugger is active)
    if "debugpy" in sys.modules:
        _original_trace = sys.gettrace()
        sys.settrace(_watch_variable_trace)
        print("=" * 60)
        print("[Visualizer] Auto-accumulation ENABLED!")
        print("=" * 60)
        print(f"Watching for variable: '{variable_name}'")
        print("Images will be accumulated (not displayed automatically)")
        print("Call show_accumulated() to display all accumulated images")
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


def show_accumulated():
    """Display all accumulated images from multiple function calls"""
    global _accumulated_images

    if not _accumulated_images:
        print("[Visualizer] No accumulated images to show")
        return

    print(f"\n[Visualizer] Displaying {len(_accumulated_images)} accumulated images")

    for i, img in enumerate(_accumulated_images):
        cv.imshow(f"Image {i + 1}/{len(_accumulated_images)}", img)
        cv.waitKey(1)

    print("[Visualizer] Press any key to close all windows...")
    cv.waitKey(0)
    cv.destroyAllWindows()

    print(f"[Visualizer] Cleared {len(_accumulated_images)} images from buffer")
    _accumulated_images = []


def clear_accumulated():
    """Clear accumulated images without showing them"""
    global _accumulated_images
    count = len(_accumulated_images)
    _accumulated_images = []
    print(f"[Visualizer] Cleared {count} accumulated images")


# Short alias for convenience
show = show_current_images  # Type just: show()
show_all = show_accumulated  # Type: show_all()
clear = clear_accumulated  # Type: clear()


# Reload helper for debugging
def reload_all():
    """
    Reload all project modules (non-standard library) while debugging.
    Call from debug console: reload_all()
    """
    import importlib
    from pathlib import Path

    # Get the workspace root (parent of src/ and dev_scripts/)
    try:
        workspace_root = Path(__file__).parent.parent.resolve()
    except:
        workspace_root = Path.cwd()

    modules_to_reload = []
    failed_reloads = []

    # Find all loaded modules that are part of this project
    for module_name, module in list(sys.modules.items()):
        if module is None:
            continue

        try:
            # Get module file path
            if not hasattr(module, "__file__") or module.__file__ is None:
                continue

            module_path = Path(module.__file__).resolve()

            # Check if module is inside workspace
            try:
                module_path.relative_to(workspace_root)
                # It's a project module, try to reload it
                try:
                    importlib.reload(module)
                    modules_to_reload.append(module_name)
                except (RuntimeError, ImportError, AttributeError, TypeError) as e:
                    # Some modules can't be reloaded (like numpy._globals)
                    failed_reloads.append((module_name, str(e)))
            except ValueError:
                # Not in workspace, skip
                pass

        except Exception:
            # Skip any problematic modules
            pass

    if modules_to_reload:
        print(f"[Reload] Reloaded {len(modules_to_reload)} module(s):")
        for mod in sorted(modules_to_reload):
            print(f"  - {mod}")
    else:
        print("[Reload] No project modules found to reload")

    if failed_reloads:
        print(
            f"[Reload] Skipped {len(failed_reloads)} module(s) that cannot be reloaded"
        )

    print("[Reload] Note: Already-defined functions won't be updated automatically")
    print("[Reload] You may need to re-import or restart for full effect")

    return modules_to_reload
