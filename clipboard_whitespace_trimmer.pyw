"""
Clipboard Whitespace Trimmer

- Removes leading and trailing whitespace from clipboard content
- Optionally removes custom characters
- Creates a system tray icon for easy access
"""

import datetime
import json
import logging
import os
import socket
import sys
import threading
import time
import tomllib
from datetime import datetime
from pathlib import Path
from PIL import Image
from pystray import Icon, MenuItem, Menu

import pyperclip

logger = logging.getLogger(__name__)

__version__ = "1.0.8"  # Major.Minor.Patch

exit_event = threading.Event()

CONFIG = {}


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
            data = tomllib.load(f)  # Replace with "toml.load(f)" if using the toml package
        return data

    except (OSError, tomllib.TOMLDecodeError):
        logger.exception(f"Failed to read TOML file: {json.dumps(str(file_path))}")
        raise


def trim_whitespaces(text: str, unwanted_characters: list) -> str:
    """
    Trim leading and trailing characters. When `unwanted_characters` is falsy,
    use str.strip() to remove all Unicode whitespace.
    """
    # Fast path: empty text
    if not text:
        return text

    # Trim from the left with emptiness guard
    while text and text[0] in unwanted_characters:
        text = text[1:]

    # Trim from the right with emptiness guard
    while text and text[-1] in unwanted_characters:
        text = text[:-1]

    return text


def load_image(path: str | Path) -> Image.Image:
    path = Path(path)
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
    logger.debug("Exit event triggered")


def startup_tray_icon():
    logger.debug("Starting up system tray icon...")
    image = load_image("system_tray_icon.png")
    menu = Menu(
        # MenuItem("Source", open_source_url),
        MenuItem("Open Folder", open_script_folder),
        MenuItem("Exit", on_exit)
    )
    icon = Icon("CenterWindowScript", image, menu=menu)
    logger.debug("Started system tray icon.")
    icon.run()


def main():
    system_tray_thread = threading.Thread(target=startup_tray_icon, daemon=True)
    system_tray_thread.start()

    unwanted_characters = CONFIG.get("unwanted_characters", [])
    logger.debug(f"Unwanted characters: {(unwanted_characters)}")

    previous_clipboard_text = None
    while True:
        try:
            current_clipboard_text = pyperclip.paste()

            # Skip empty reads
            if not current_clipboard_text:
                time.sleep(0.1)
                continue

            # Skip unchanged content
            if current_clipboard_text == previous_clipboard_text:
                time.sleep(0.1)
                continue

            # Detect if any of the unwanted characters are present at the beginning or end of current_clipboard_text
            if current_clipboard_text[0] in unwanted_characters or current_clipboard_text[-1] in unwanted_characters:
                logger.info(f"Extra white spaces detected in {json.dumps(str(current_clipboard_text))}")
                cleaned_text = trim_whitespaces(current_clipboard_text, unwanted_characters)
                pyperclip.copy(cleaned_text)
                logger.info(f"Clipboard updated to {json.dumps(str(cleaned_text))}")
                previous_clipboard_text = cleaned_text
                continue

        except pyperclip.PyperclipException:
            logger.exception("Clipboard access error")
        except Exception:  # pylint: disable=broad-except
            logger.exception("An unknown error occurred")

        # Debounce
        time.sleep(0.1)


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
        global CONFIG
        CONFIG = load_config(config_path)
        logger_config = CONFIG.get("logging", {})

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

        exit_behavior_config = CONFIG.get("exit_behavior", {})
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
        # Using "err" or "exc" is standard; logging the traceback handles the "broad-except"
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
