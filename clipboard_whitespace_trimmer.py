import datetime
import logging
import os
import pathlib
import pyperclip
import socket
import sys
import threading
import time
import toml
import traceback
import typing
from datetime import datetime
from PIL import Image
from pystray import Icon, MenuItem, Menu

logger = logging.getLogger(__name__)

"""
Clipboard Whitespace Trimmer

- Removes leading and trailing whitespace from clipboard content
- Optionally removes custom characters
- Creates a system tray icon for easy access
"""

__version__ = "1.0.5"  # Major.Minor.Patch


exit_event = threading.Event()


def read_toml(file_path: typing.Union[str, pathlib.Path]) -> dict:
    """
    Read configuration settings from the TOML file.
    """
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f'File not found: "{file_path}"')
    config = toml.load(file_path)
    return config


def trim_whitespaces(text: str, unwanted_characters: list) -> str:
    """
    Use str.strip for trimming. If custom characters are provided,
    build a character string for strip; otherwise trim all Unicode whitespace.
    """
    logger.debug('Trimming...')
    if not text:
        logger.debug('Input text is empty. Nothing to trim.')
        return text

    if not unwanted_characters:
        # Trim all Unicode whitespace
        cleaned = text.strip()
    else:
        # str.strip accepts a string of characters to trim
        chars = ''.join(unwanted_characters)
        cleaned = text.strip(chars)

    return cleaned


def load_image(path) -> Image.Image:
    image = Image.open(path)
    logger.debug(f'Loaded image at path "{path}"')
    return image


# def open_source_url(icon, item):
#     webbrowser.open("https://github.com/RandomGgames/Window-Centerer")
#     logger.debug('Opened source URL.')


def open_script_folder(icon, item):
    folder_path = os.path.dirname(os.path.abspath(__file__))
    os.startfile(folder_path)
    logger.debug(f'Opened script folder: {folder_path}')


def on_exit(icon, item):
    logger.debug(f'Exit pressed on system tray icon')
    icon.stop()
    logger.debug('System tray icon stopped.')
    exit_event.set()
    logger.debug(f'exit event triggered')


def startup_tray_icon():
    logger.debug(f'Starting up system tray icon')
    image = load_image('system_tray_icon.png')
    menu = Menu(
        # MenuItem('Source', open_source_url),
        MenuItem('Open Folder', open_script_folder),
        MenuItem('Exit', on_exit)
    )
    icon = Icon('CenterWindowScript', image, menu=menu)
    logger.debug(f'Started system tray icon')
    icon.run()


def main():
    system_tray_thread = threading.Thread(target=startup_tray_icon, daemon=True)
    system_tray_thread.start()

    previous_clipboard_text = ''
    unwanted_characters = config.get("unwanted_characters", [])

    # Precompute the strip characters string for detection efficiency
    strip_chars = ''.join(unwanted_characters) if unwanted_characters else None

    # Polling loop
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

            # Detection using strip rather than indexing ends
            candidate_cleaned = (
                current_clipboard_text.strip()
                if strip_chars is None
                else current_clipboard_text.strip(strip_chars)
            )

            if candidate_cleaned != current_clipboard_text:
                logger.info(f'Extra white spaces detected in {repr(current_clipboard_text)}')
                cleaned_text = trim_whitespaces(current_clipboard_text, unwanted_characters)
                pyperclip.copy(cleaned_text)
                logger.info(f'Clipboard updated to {repr(cleaned_text)}')
                previous_clipboard_text = cleaned_text
            else:
                logger.info(f'No extra white spaces detected in {repr(current_clipboard_text)}')
                previous_clipboard_text = current_clipboard_text

        except pyperclip.PyperclipException as e:
            logger.exception(f'Clipboard access error: {repr(e)}')
        except Exception as e:
            logger.exception(f'An error occurred due to {repr(e)}')

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
        ('y', 365 * 24 * 60 * 60 * 1_000_000_000),
        ('mo', 30 * 24 * 60 * 60 * 1_000_000_000),
        ('d', 24 * 60 * 60 * 1_000_000_000),
        ('h', 60 * 60 * 1_000_000_000),
        ('m', 60 * 1_000_000_000),
        ('s', 1_000_000_000),
        ('ms', 1_000_000),
        ('us', 1_000),
        ('ns', 1),
    ]
    parts = []
    for name, factor in units:
        value, ns = divmod(ns, factor)
        if value:
            parts.append(f'{value}{name}')
        if len(parts) == 2:
            break
    if not parts:
        return "0s"
    return "".join(parts)


def enforce_max_folder_size(log_dir: pathlib.Path, max_bytes: int) -> None:
    """
    Enforce a maximum total size for all logs in the folder.
    Deletes oldest logs until below limit.
    """
    if max_bytes is None:
        return

    files = sorted(
        [f for f in log_dir.glob("*.log*") if f.is_file()],
        key=lambda f: f.stat().st_mtime
    )

    total_size = sum(f.stat().st_size for f in files)

    while total_size > max_bytes and files:
        oldest = files.pop(0)
        try:
            size = oldest.stat().st_size
            oldest.unlink()
            logger.debug(f'Deleted "{oldest}"')
            total_size -= size
        except Exception:
            logger.error(f'Failed to delete "{oldest}"', exc_info=True)
            continue


def setup_logging(
        logger: logging.Logger,
        log_file_path: typing.Union[str, pathlib.Path],
        max_folder_size_bytes: typing.Union[int, None] = None,
        console_logging_level: int = logging.DEBUG,
        file_logging_level: int = logging.DEBUG,
        log_message_format: str = "%(asctime)s.%(msecs)03d %(levelname)s [%(funcName)s]: %(message)s",
        date_format: str = "%Y-%m-%d %H:%M:%S"
) -> None:

    log_file_path = pathlib.Path(log_file_path)
    log_dir = log_file_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.handlers.clear()
    logger.setLevel(file_logging_level)

    formatter = logging.Formatter(log_message_format, datefmt=date_format)

    # File Handler
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(file_logging_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_logging_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if max_folder_size_bytes is not None:
        enforce_max_folder_size(log_dir, max_folder_size_bytes)


def load_config(file_path: typing.Union[str, pathlib.Path]) -> dict:
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f'File not found: "{file_path}"')
    config = read_toml(file_path)
    return config


if __name__ == "__main__":
    error = 0
    try:
        script_name = pathlib.Path(__file__).stem
        config_path = pathlib.Path(f'{script_name}_config.toml')
        # config_path = pathlib.Path("config.toml")
        config = load_config(config_path)

        logging_config = config.get("logging", {})
        console_logging_level = getattr(logging, logging_config.get("console_logging_level", "INFO").upper(), logging.DEBUG)
        file_logging_level = getattr(logging, logging_config.get("file_logging_level", "INFO").upper(), logging.DEBUG)
        log_message_format = logging_config.get("log_message_format", "%(asctime)s.%(msecs)03d %(levelname)s [%(funcName)s]: %(message)s")
        logs_folder_name = logging_config.get("logs_folder_name", "logs")
        max_folder_size_bytes = logging_config.get("max_folder_size", None)

        pc_name = socket.gethostname()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = pathlib.Path(logs_folder_name) / script_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_name = f'{timestamp}_{script_name}_{pc_name}.log'
        log_file_path = log_dir / log_file_name

        setup_logging(
            logger,
            log_file_path,
            max_folder_size_bytes=max_folder_size_bytes,
            console_logging_level=console_logging_level,
            file_logging_level=file_logging_level,
            log_message_format=log_message_format
        )
        start_time = time.perf_counter_ns()
        logger.info(f'Script: "{script_name}" | Version: {__version__} | Host: "{pc_name}"')
        main()
        end_time = time.perf_counter_ns()
        duration = end_time - start_time
        duration = format_duration_long(duration / 1e9)
        logger.info(f'Execution completed in {duration}.')
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user.")
        error = 130
    except Exception as e:
        logger.warning(f'A fatal error has occurred: {repr(e)}\n{traceback.format_exc()}')
        error = 1
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()
        input("Press Enter to exit...")
        sys.exit(error)
