# Clipboard Whitespace Trimmer

Small Windows utility that monitors the clipboard and automatically trims unwanted whitespace characters (tabs, line breaks, spaces, etc.) from copied text. Runs as a background tray application and writes logs to the `logs/` folder.

## Features
- Monitors system clipboard and strips configured unwanted characters
- Lightweight system tray app (uses an icon `system_tray_icon.png`)
- Configurable via `clipboard_whitespace_trimmer_config.toml`
- Logs to `logs/` with adjustable verbosity

## Requirements
- Python 3.8+
- See `requirements.txt` for Python package dependencies (pillow, pyperclip, pystray)

## Installation
1. Clone or download this repository.
2. From a command prompt in the repository folder, install dependencies:

```cmd
pip install -r requirements.txt
```

## Configuration
The behavior is controlled by `clipboard_whitespace_trimmer_config.toml` in the project root.

Key options:
- `unwanted_characters`: array of characters to remove from clipboard text (default contains `"\t", "\r", "\n", " "`).
- `[logging]`: set console/file logging levels and formatting.
- `[exit_behavior]`: control pause-on-exit behavior for debugging.

Edit the TOML file to suit your needs and save before starting the app.

## Usage
- To run in a visible console (useful for debugging):

```cmd
python clipboard_whitespace_trimmer.pyw
```

- To run in the background on Windows (no console window):

```cmd
pythonw clipboard_whitespace_trimmer.pyw
```

When running, the app places an icon in the system tray. Use the tray menu to pause/resume or quit.

## Logs
Logs are written to the `logs/` directory. The default configuration keeps up to `max_log_files` files. Adjust logging levels in the TOML config.

## Troubleshooting
- Nothing happens when I copy text:
  - Ensure Python dependencies are installed.
  - Start with the console (`python ...`) to see DEBUG output.
- Clipboard modifications not desired:
  - Edit `unwanted_characters` in the config and restart the app.
- Tray icon missing or app won't start:
  - Confirm `system_tray_icon.png` is present in the repository root.

## Contributing
Small fixes, bug reports, and config improvements are welcome. Open issues or PRs against the repository.
