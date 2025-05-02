"""Logging utilities."""

import logging
import pathlib
import sys
from typing import Optional

from mpi4py import MPI

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

COLORS = {
    "WARNING": YELLOW,
    "INFO": WHITE,
    "DEBUG": GREEN,
    "CRITICAL": RED,
    "ERROR": RED,
}


class ColoredFormatter(logging.Formatter):
    """A logging formatter that adds colors to specific parts of the log message."""

    def format(self, record: logging.LogRecord) -> str:
        """Apply color formating to log message."""
        levelname = record.levelname
        color_level = COLOR_SEQ % (30 + COLORS.get(levelname, 0))
        record.levelname = f"{color_level}{levelname}{RESET_SEQ}"

        # Add colors to specific fields
        record.asctime = (
            f"{COLOR_SEQ % (30 + GREEN)}{self.formatTime(record)}{RESET_SEQ}"
        )
        record.name = f"{COLOR_SEQ % (30 + CYAN)}{record.name}{RESET_SEQ}"
        record.funcName = f"{COLOR_SEQ % (30 + MAGENTA)}{record.funcName}{RESET_SEQ}"
        record.msg = f"{color_level}{record.msg}{RESET_SEQ}"

        return super().format(record)


def set_logger_config(
    level: int = logging.INFO,
    log_file: Optional[str | pathlib.Path] = None,
    log_to_stdout: bool = True,
    log_rank: bool = False,
    colors: bool = True,
) -> None:
    """
    Configure the logging settings for the application.

    Parameters
    ----------
    level : int, optional
        The logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.
    log_file : str or pathlib.Path, optional
        Path to the log file where logs should be written. If None, logs will not be
        written to a file. Defaults to None.
    log_to_stdout : bool, optional
        Whether to log messages to the standard output. Defaults to True.
    log_rank : bool, optional
        Whether to include the MPI rank in the log messages. Defaults to False.
    colors : bool, optional
        Whether to use colored log messages in the console output. Defaults to True.

    Returns
    -------
    None
    """
    rank = (
        f"R{MPI.COMM_WORLD.Get_rank()}/{MPI.COMM_WORLD.Get_size()}:" if log_rank else ""
    )
    base_logger = logging.getLogger("tensorcraft")
    format_string = (
        f"[%(asctime)s][%(name)s][%(funcName)s][%(levelname)s] - {rank}%(message)s"
    )

    simple_formatter = logging.Formatter(fmt=format_string)

    # Std out
    if log_to_stdout:
        std_handler = logging.StreamHandler(stream=sys.stdout)
        if colors:
            formatter = ColoredFormatter(fmt=format_string)
            std_handler.setFormatter(formatter)
        else:
            std_handler.setFormatter(simple_formatter)

        base_logger.addHandler(std_handler)

    if log_file is not None:
        log_file = pathlib.Path(log_file)
        log_dir = log_file.parents[0]
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(simple_formatter)
        base_logger.addHandler(file_handler)

    base_logger.setLevel(level)
