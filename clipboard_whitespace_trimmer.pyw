"""
Clipboard Whitespace Trimmer

- Removes leading and trailing whitespace from clipboard content
- Optionally removes custom characters
- Creates a system tray icon for easy access
"""

import ctypes
import datetime
import json
import logging
import os
import socket
import sys
import threading
import time
import tomllib
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from pystray import Icon, MenuItem, Menu

logger = logging.getLogger(__name__)

__version__ = "1.0.9"  # Major.Minor.Patch

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# user32
user32.OpenClipboard.argtypes = [wintypes.HWND]
user32.OpenClipboard.restype = wintypes.BOOL

user32.CloseClipboard.argtypes = []
user32.CloseClipboard.restype = wintypes.BOOL

user32.EnumClipboardFormats.argtypes = [wintypes.UINT]
user32.EnumClipboardFormats.restype = wintypes.UINT

user32.GetClipboardData.argtypes = [wintypes.UINT]
user32.GetClipboardData.restype = wintypes.HANDLE

# kernel32
kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalLock.restype = wintypes.LPVOID

kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalUnlock.restype = wintypes.BOOL

kernel32.GlobalSize.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalSize.restype = ctypes.c_size_t


# Function signatures
user32.OpenClipboard.argtypes = [wintypes.HWND]
user32.OpenClipboard.restype = wintypes.BOOL
user32.EmptyClipboard.argtypes = []
user32.EmptyClipboard.restype = wintypes.BOOL
user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
user32.SetClipboardData.restype = wintypes.HANDLE
user32.CloseClipboard.argtypes = []
user32.CloseClipboard.restype = wintypes.BOOL

kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalLock.restype = wintypes.LPVOID
kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalUnlock.restype = wintypes.BOOL

GMEM_MOVEABLE = 0x0002


exit_event = threading.Event()


def read_toml(file_path: Path | str) -> dict:
    """
    Reads a TOML file and returns its contents as a dictionary.

    Args:
        file_path (Path | str): The file path of the TOML file to read.

    Returns:
        dict: The contents of the TOML file as a dictionary.

    Raises:
        FileNotFoundError: If the TOML file does not exist.
        OSError: If the file cannot be read.
        tomllib.TOMLDecodeError (or toml.TomlDecodeError): If the file is invalid TOML.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {json.dumps(str(path))}")

    try:
        # Read TOML as bytes
        with path.open("rb") as f:
            data = tomllib.load(f)  # Replace with 'toml.load(f)' if using the toml package
        return data

    except (OSError, tomllib.TOMLDecodeError):
        logger.exception(f"Failed to read TOML file: {json.dumps(str(file_path))}")
        raise


def trim_whitespaces(text: str, unwanted_characters: list) -> str:
    """
    Trim leading and trailing characters. When `unwanted_characters` is falsy,
    use str.strip() to remove all Unicode whitespace.
    """
    logger.debug(f"Trimming {repr(text)}")

    # Fast path: empty text
    if not text:
        logger.debug("Input text is empty. Nothing to trim.")
        return text

    # Trim from the left with emptiness guard
    while text and text[0] in unwanted_characters:
        logger.debug(f"Removing leading character {repr(text[0])}")
        text = text[1:]
        logger.debug(f"Result: {repr(text)}")

    # Trim from the right with emptiness guard
    while text and text[-1] in unwanted_characters:
        logger.debug(f"Removing trailing character {repr(text[-1])}")
        text = text[:-1]
        logger.debug(f"Result: {repr(text)}")

    return text


def load_image(path: Path | str) -> Image.Image:
    image = Image.open(path)
    logger.debug(f"Loaded image at path {json.dumps(str(path))}")
    return image


# def open_source_url(icon, item):
#     webbrowser.open("https://github.com/RandomGgames/Window-Centerer")
#     logger.debug("Opened source URL.")


def open_script_folder():
    folder_path = os.path.dirname(os.path.abspath(__file__))
    os.startfile(folder_path)
    logger.debug(f"Opened script folder: {json.dumps(str(folder_path))}")


def on_exit(icon):
    logger.debug("Exit pressed on system tray icon")
    icon.stop()
    logger.debug("System tray icon stopped.")
    exit_event.set()
    logger.debug("exit event triggered")


def startup_tray_icon():
    logger.debug("Starting up system tray icon")
    image = load_image("system_tray_icon.png")
    menu = Menu(
        # MenuItem("Source", open_source_url),
        MenuItem("Open Folder", open_script_folder),
        MenuItem("Exit", on_exit)
    )
    icon = Icon("CenterWindowScript", image, menu=menu)
    logger.debug("Started system tray icon")
    icon.run()


def open_clipboard_with_retry(retries: int = 5, delay: float = 0.02) -> bool:
    """
    Attempt to open the clipboard with retries.
    Returns True if successful, False otherwise.
    """
    for _ in range(retries):
        if user32.OpenClipboard(None):
            return True
        time.sleep(delay)
    return False


def get_clipboard_all_data() -> dict[int, str | bytes]:
    """
    Return all clipboard formats as {format_id: data}.
    Text formats are decoded to str.
    All other formats are raw bytes.
    """
    CF_TEXT = 1
    CF_UNICODETEXT = 13

    clipboard_data: dict[int, str | bytes] = {}

    if not open_clipboard_with_retry():
        logger.debug("Clipboard is busy; skipping this cycle")
        return clipboard_data

    try:
        fmt = 0
        while True:
            fmt = user32.EnumClipboardFormats(fmt)
            if fmt == 0:
                break

            handle = user32.GetClipboardData(fmt)
            if not handle:
                continue

            ptr = kernel32.GlobalLock(handle)
            if not ptr:
                continue

            try:
                size = kernel32.GlobalSize(handle)
                raw = ctypes.string_at(ptr, size)

                if fmt == CF_UNICODETEXT:
                    text = raw.decode("utf-16le", errors="replace").rstrip("\x00")
                    clipboard_data[fmt] = normalize_unicode_text(text)

                elif fmt == CF_TEXT:
                    codepage = ctypes.windll.kernel32.GetACP()
                    clipboard_data[fmt] = (
                        raw.split(b"\x00", 1)[0]
                        .decode(f"cp{codepage}", errors="replace")
                    )

                else:
                    clipboard_data[fmt] = raw

            finally:
                kernel32.GlobalUnlock(handle)

    finally:
        user32.CloseClipboard()

    return clipboard_data


def normalize_unicode_text(text: str) -> str:
    """
    Normalize CF_UNICODETEXT coming from the clipboard.

    Excel (and some other apps) prepend CR/LF even when the cell has
    no visible leading newline. Remove only leading newlines.
    """
    return text.lstrip("\r\n")


def write_json_file(data: dict | list, file_path: Path | str) -> None:
    """
    Writes a dictionary or list to a json file. Includes error checking and logging.

    Args:
    data (dict | list): The data to write to the json file.
    file_path (Path | str): The file path of the json file to write.

    Returns:
    None
    """
    try:
        file_path = Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created folder: {json.dumps(str(file_path.parent))}")
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
            logger.info(f"Successfully wrote json file: {json.dumps(str(file_path))}")
    except IOError:
        logger.error(f"Error writing json file: {json.dumps(str(file_path))}")
        raise


def json_safe_clipboard(data: dict[int, str | bytes]) -> dict[str, Any]:
    """
    Convert clipboard data into JSON-serializable form.
    Bytes are stored as their Python repr (e.g. b'\\x00\\xff').
    """
    safe: dict[str, Any] = {}

    for fmt, value in data.items():
        key = str(fmt)

        if isinstance(value, bytes):
            safe[key] = repr(value)
        else:
            safe[key] = value

    return safe


def trim_clipboard_text(text: str, unwanted_characters=None) -> str:
    """
    Trim leading and trailing characters from clipboard text.
    By default, removes: tab, carriage return, newline, space.
    """
    if unwanted_characters is None:
        unwanted_characters = ["\t", "\r", "\n", " "]

    # Trim only leading/trailing chars
    start = 0
    end = len(text)

    while start < end and text[start] in unwanted_characters:
        start += 1
    while end > start and text[end - 1] in unwanted_characters:
        end -= 1

    return text[start:end]


def replace_clipboard_with_trimmed_text():
    """
    Open the clipboard, read all formats, trim text formats, then
    write back all formats (text trimmed, everything else untouched).
    """
    CF_TEXT = 1
    CF_UNICODETEXT = 13

    if not open_clipboard_with_retry():
        logger.debug("Clipboard busy, cannot update")
        return False

    try:
        # Read all clipboard formats
        formats = {}
        fmt = 0
        while True:
            fmt = user32.EnumClipboardFormats(fmt)
            if fmt == 0:
                break

            handle = user32.GetClipboardData(fmt)
            if not handle:
                continue

            ptr = kernel32.GlobalLock(handle)
            if not ptr:
                continue

            try:
                size = kernel32.GlobalSize(handle)
                raw = ctypes.string_at(ptr, size)

                # Decode text formats, trim, re-encode
                if fmt == CF_UNICODETEXT:
                    text = raw.decode("utf-16le", errors="replace").rstrip("\x00")
                    trimmed = trim_clipboard_text(text)
                    formats[fmt] = (trimmed + "\x00").encode("utf-16le")

                elif fmt == CF_TEXT:
                    codepage = ctypes.windll.kernel32.GetACP()
                    text = raw.split(b"\x00", 1)[0].decode(f"cp{codepage}", errors="replace")
                    trimmed = trim_clipboard_text(text)
                    formats[fmt] = trimmed.encode(f"cp{codepage}") + b"\x00"

                else:
                    # Keep raw bytes untouched
                    formats[fmt] = raw

            finally:
                kernel32.GlobalUnlock(handle)

        # Now write back all formats
        user32.EmptyClipboard()
        for fmt, data in formats.items():
            hGlobal = kernel32.GlobalAlloc(0x2000, len(data))  # GMEM_MOVEABLE
            ptr = kernel32.GlobalLock(hGlobal)
            ctypes.memmove(ptr, data, len(data))
            kernel32.GlobalUnlock(hGlobal)
            user32.SetClipboardData(fmt, hGlobal)

        logger.debug("Clipboard text trimmed and replaced successfully")
        return True

    finally:
        user32.CloseClipboard()


def trim_clipboard_text_formats(clipboard: dict[int, str | bytes], unwanted_characters: list[str]) -> dict[int, str | bytes]:
    """
    Trim leading/trailing unwanted characters for specific text formats using match-case.
    Other formats are left untouched.
    """
    new_clipboard = {}
    unwanted_set = set(unwanted_characters)

    for fmt, value in clipboard.items():
        match fmt:
            case 1 | 13 | 7:  # CF_TEXT, CF_UNICODETEXT, legacy RTF/text
                if isinstance(value, str):
                    # Trim leading
                    start = 0
                    while start < len(value) and value[start] in unwanted_set:
                        start += 1
                    # Trim trailing
                    end = len(value)
                    while end > start and value[end - 1] in unwanted_set:
                        end -= 1
                    new_clipboard[fmt] = value[start:end]
                else:
                    new_clipboard[fmt] = value
            case _:  # all other formats
                new_clipboard[fmt] = value

    return new_clipboard


def set_clipboard_data(clipboard_data: dict[int, str | bytes]) -> None:
    """
    Set multiple formats to the Windows clipboard.
    Text is automatically encoded if needed; bytes are written as-is.
    """
    if not user32.OpenClipboard(None):
        raise RuntimeError("Cannot open clipboard")

    try:
        user32.EmptyClipboard()

        for fmt, value in clipboard_data.items():
            # Determine the data type to write
            if isinstance(value, str):
                # Convert string to bytes depending on format
                if fmt == 13:  # CF_UNICODETEXT
                    data_bytes = (value + "\x00").encode("utf-16le")
                elif fmt == 1:  # CF_TEXT
                    data_bytes = (value + "\x00").encode("mbcs")
                else:
                    data_bytes = (value + "\x00").encode("utf-8")
            else:
                data_bytes = value

            # Allocate global memory
            h_global = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data_bytes))
            if not h_global:
                raise MemoryError("Failed to allocate global memory for clipboard")

            ptr = kernel32.GlobalLock(h_global)
            if not ptr:
                raise MemoryError("Failed to lock global memory for clipboard")

            try:
                ctypes.memmove(ptr, data_bytes, len(data_bytes))
            finally:
                kernel32.GlobalUnlock(h_global)

            if not user32.SetClipboardData(fmt, h_global):
                raise RuntimeError(f"Failed to set clipboard format {fmt}")

    finally:
        user32.CloseClipboard()


def main():
    system_tray_thread = threading.Thread(target=startup_tray_icon, daemon=True)
    system_tray_thread.start()

    unwanted_characters = config.get("unwanted_characters", [])
    logger.debug(f"Unwanted characters: {unwanted_characters}")

    previous_clipboard = None

    while not exit_event.is_set():
        clipboard = get_clipboard_all_data()
        if not clipboard:
            time.sleep(0.5)
            # logger.debug("Clipboard is empty; skipping this cycle")
            continue

        if previous_clipboard == clipboard:
            # logger.debug("Clipboard data has not changed; skipping this cycle")
            time.sleep(0.5)
            continue

        cleaned_clipboard = trim_clipboard_text_formats(clipboard, unwanted_characters)
        set_clipboard_data(cleaned_clipboard)

        previous_clipboard = clipboard

        time.sleep(0.5)


def format_duration_long(duration_seconds: float) -> str:
    """
    Format duration in a human-friendly way, showing only the two largest non-zero units.
    For durations >= 1s, do not show microseconds or nanoseconds.
    For durations >= 1m, do not show milliseconds.
    """
    ns = int(duration_seconds * 1_000_000_000)
    units = [
        ("y", 365 * 24 * 60 * 60 * 1_000_000_000),
        ("mo", 30 * 24 * 60 * 60 * 1_000_000_000),
        ("d", 24 * 60 * 60 * 1_000_000_000),
        ("h", 60 * 60 * 1_000_000_000),
        ("m", 60 * 1_000_000_000),
        ("s", 1_000_000_000),
        ("ms", 1_000_000),
        ("us", 1_000),
        ("ns", 1),
    ]
    parts = []
    for name, factor in units:
        value, ns = divmod(ns, factor)
        if value:
            parts.append(f"{value}{name}")
        if len(parts) == 2:
            break
    if not parts:
        return "0s"
    return "".join(parts)


def enforce_max_log_count(dir_path: Path | str, max_count: int | None, script_name: str) -> None:
    """Keep only the N most recent logs for this script."""
    if max_count is None or max_count <= 0:
        return

    dir_path = Path(dir_path)

    # Get all logs for this script, sorted by name (which is our timestamp)
    # Newest will be at the end of the list
    files = sorted([f for f in dir_path.glob(f"*{script_name}*.log") if f.is_file()])

    # If we have more than the limit, calculate how many to delete
    if len(files) > max_count:
        to_delete = files[:-max_count]  # Everything except the last N files
        for f in to_delete:
            try:
                f.unlink()
                logger.debug(f"Deleted old log: {f.name}")
            except OSError as e:
                logger.error(f"Failed to delete {f.name}: {e}")


def setup_logging(
        logger_obj: logging.Logger,
        file_path: Path | str,
        script_name: str,
        max_log_files: int | None = None,
        console_logging_level: int = logging.DEBUG,
        file_logging_level: int = logging.DEBUG,
        message_format: str = "%(asctime)s.%(msecs)03d %(levelname)s [%(funcName)s]: %(message)s",
        date_format: str = "%Y-%m-%d %H:%M:%S"
) -> None:
    """
    Set up logging for a script.

    Args:
    logger_obj (logging.Logger): The logger object to configure.
    file_path (Path | str): The file path of the log file to write.
    max_log_files (int | None, optional): The maximum total size for all logs in the folder. Defaults to None.
    console_logging_level (int, optional): The logging level for console output. Defaults to logging.DEBUG.
    file_logging_level (int, optional): The logging level for file output. Defaults to logging.DEBUG.
    message_format (str, optional): The format string for log messages. Defaults to "%(asctime)s.%(msecs)03d %(levelname)s [%(funcName)s]: %(message)s".
    date_format (str, optional): The format string for log timestamps. Defaults to "%Y-%m-%d %H:%M:%S".
    """

    file_path = Path(file_path)
    dir_path = file_path.parent
    dir_path.mkdir(parents=True, exist_ok=True)

    logger_obj.handlers.clear()
    logger_obj.setLevel(file_logging_level)

    formatter = logging.Formatter(message_format, datefmt=date_format)

    # File Handler
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(file_logging_level)
    file_handler.setFormatter(formatter)
    logger_obj.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_logging_level)
    console_handler.setFormatter(formatter)
    logger_obj.addHandler(console_handler)

    if max_log_files is not None:
        enforce_max_log_count(dir_path, max_log_files, script_name)


def load_config(file_path: Path | str) -> dict:
    """
    Load configuration from a TOML file.

    Args:
    file_path (Path | str): The file path of the TOML file to read.

    Returns:
    dict: The contents of the TOML file as a dictionary.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {json.dumps(str(file_path))}")
    data = read_toml(file_path)
    return data


def bootstrap():
    """
    Handles environment setup, configuration loading,
    and logging before executing the main script logic.
    """
    exit_code = 0
    try:
        # Resolve paths and configuration
        script_path = Path(__file__)
        script_name = script_path.stem
        config_path = script_path.with_name(f"{script_name}_config.toml")

        # Load settings
        global config
        config = load_config(config_path)
        logger_config = config.get("logging", {})

        # Parse log levels and formats
        console_log_level = getattr(logging, logger_config.get("console_logging_level", "INFO").upper(), logging.INFO)
        file_log_level = getattr(logging, logger_config.get("file_logging_level", "INFO").upper(), logging.INFO)
        log_message_format = logger_config.get("log_message_format", "%(asctime)s.%(msecs)03d %(levelname)s [%(funcName)s] - %(message)s")

        # Setup directories and filenames
        logs_folder = Path(logger_config.get("logs_folder_name", "logs"))
        logs_folder.mkdir(parents=True, exist_ok=True)

        pc_name = socket.gethostname()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_folder / f"{timestamp}__{script_name}__{pc_name}.log"

        # Initialize logging
        setup_logging(
            logger_obj=logger,
            file_path=log_path,
            script_name=script_name,
            max_log_files=logger_config.get("max_log_files"),
            console_logging_level=console_log_level,
            file_logging_level=file_log_level,
            message_format=log_message_format
        )

        exit_behavior_config = config.get("exit_behavior", {})
        pause_before_exit = exit_behavior_config.get("always_pause", False)
        pause_before_exit_on_error = exit_behavior_config.get("pause_on_error", True)

        start_ns = time.perf_counter_ns()
        logger.info(f"Script: {json.dumps(script_name)} | Version: {__version__} | Host: {json.dumps(pc_name)}")

        main()

        end_ns = time.perf_counter_ns()
        duration_str = format_duration_long((end_ns - start_ns) / 1e9)
        logger.info(f"Execution completed in {duration_str}.")

    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user.")
        exit_code = 130
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Using 'err' or 'exc' is standard; logging the traceback handles the 'broad-except'
        logger.error(f"A fatal error has occurred: {e}")
        exit_code = 1
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    if pause_before_exit or (pause_before_exit_on_error and exit_code != 0):
        input("Press Enter to exit...")

    return exit_code


if __name__ == "__main__":
    sys.exit(bootstrap())
