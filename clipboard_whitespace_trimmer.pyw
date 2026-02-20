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
import pyperclip
import send2trash
import socket
import sys
import threading
import time
import tomllib
import webbrowser
from datetime import datetime
from pathlib import Path
from PIL import Image
from pystray import Icon, MenuItem, Menu

logger = logging.getLogger(__name__)

__version__ = "1.0.11"  # Major.Minor.Patch

exit_event = threading.Event()
running_event = threading.Event()
running_event.set()

CONFIG = {}


def load_image(path: str | Path) -> Image.Image:
    path = Path(path)
    image = Image.open(path)
    logger.debug(f"Loaded image at path {json.dumps(str(path))}")
    return image


def open_source_url():
    webbrowser.open("https://github.com/RandomGgames/clipboard_whitespace_trimmer.py")
    logger.debug("Opened source URL.")


def open_issues_url():
    webbrowser.open("https://github.com/RandomGgames/clipboard_whitespace_trimmer.py/issues")
    logger.debug("Opened issues URL.")


def open_script_folder():
    folder_path = os.path.dirname(os.path.abspath(__file__))
    os.startfile(folder_path)
    logger.debug(f"Opened script folder: {json.dumps(str(folder_path))}")


def toggle_pause(icon):
    """Toggle pause state."""
    if running_event.is_set():
        running_event.clear()
        logger.info("Paused clipboard monitor.")
    else:
        running_event.set()
        logger.info("Resumed clipboard monitor.")

    icon.update_menu()


def pause_checked(_):
    """Return True when paused."""
    return not running_event.is_set()


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
        MenuItem(f"Clipboard Whitespace Trimmer v{__version__}", None, enabled=False),
        Menu.SEPARATOR,
        MenuItem("Pause Clipboard Monitor", toggle_pause, checked=pause_checked),
        Menu.SEPARATOR,
        MenuItem("Source Page", open_source_url),
        MenuItem("Issues Page", open_issues_url),
        MenuItem("Open File Path", open_script_folder),
        Menu.SEPARATOR,
        MenuItem("Exit", on_exit),
    )

    icon = Icon("Clipboard Whitespace Trimmer", image, menu=menu)

    logger.debug("Started system tray icon.")
    icon.run()


def main():
    system_tray_thread = threading.Thread(target=startup_tray_icon, daemon=True)
    system_tray_thread.start()

    unwanted_characters = CONFIG.get("unwanted_characters", ["\n", "\t", " ", "\r"])
    logger.debug(f"Unwanted characters: {unwanted_characters}")
    unwanted_characters = "".join(unwanted_characters)
    logger.debug(f"{unwanted_characters=}")

    previous_clipboard_text = None
    while not exit_event.is_set():
        running_event.wait()

        try:
            current_clipboard_text = pyperclip.paste()

            if current_clipboard_text != previous_clipboard_text:
                logger.debug("Clipboard update detected...")

                logger.debug("Detecting unwanted characters...")
                cleaned_clipboard = current_clipboard_text.strip(unwanted_characters)

                if cleaned_clipboard != current_clipboard_text:
                    logger.debug("Detected unwanted characters in clipboard...")
                    pyperclip.copy(cleaned_clipboard)
                    logger.info("Clipboard updated with trimmed version.")
                    previous_clipboard_text = cleaned_clipboard
                else:
                    logger.info("No unwanted leading/trailing characters detected.")
                    previous_clipboard_text = current_clipboard_text
        except pyperclip.PyperclipException:
            logger.exception("Clipboard access error")
        except Exception:
            logger.exception("An unknown error occurred")

        time.sleep(1)

    logger.info("Exit event detected. Shutting down main loop.")


def enforce_max_log_count(dir_path: Path | str, max_count: int | None, script_name: str) -> None:
    """
    Keep only the N most recent logs for this script.

    Args:
        dir_path (Path | str): The directory path to the log files.
        max_count (int | None): The maximum number of log files to keep. None for no limit.
        script_name (str): The name of the script to filter logs by.
    """
    if max_count is None or max_count <= 0:
        return
    dir_path = Path(dir_path)
    files = sorted([f for f in dir_path.glob(f"*{script_name}*.log") if f.is_file()])  # Newest will be at the end of the list
    if len(files) > max_count:
        to_delete = files[:-max_count]  # Everything except the last N files
        for f in to_delete:
            try:
                send2trash.send2trash(f)
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
        script_path = Path(__file__)
        script_name = script_path.stem
        pc_name = socket.gethostname()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        config_path = script_path.with_name(f"{script_name}_config.toml")
        global CONFIG
        CONFIG = load_config(config_path)
        logger_config = CONFIG.get("logging", {})
        console_log_level = getattr(logging, logger_config.get("console_logging_level", "INFO").upper(), logging.INFO)
        file_log_level = getattr(logging, logger_config.get("file_logging_level", "INFO").upper(), logging.INFO)
        log_message_format = logger_config.get("log_message_format", "%(asctime)s.%(msecs)03d %(levelname)s [%(funcName)s] - %(message)s")
        logs_folder = Path(logger_config.get("logs_folder_name", "logs"))
        log_path = logs_folder / f"{timestamp}__{script_name}__{pc_name}.log"
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

        logger.info(f"Script: {json.dumps(script_name)} | Version: {__version__} | Host: {json.dumps(pc_name)}")
        main()

    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user.")
        exit_code = 130
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Using 'err' or 'exc' is standard; logging the traceback handles the 'broad-except'
        logger.error(f"A fatal error has occurred: {e}")
        exit_code = 1
    finally:
        logger.info("Closing loggers and existing.")
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    if pause_before_exit or (pause_before_exit_on_error and exit_code != 0):
        input("Press Enter to exit...")

    return exit_code


if __name__ == "__main__":
    sys.exit(bootstrap())
