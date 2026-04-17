from __future__ import annotations

import logging

from rich.logging import RichHandler


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=True,
                show_path=verbose,
                show_time=True,
            )
        ],
        force=True,
    )
    # Quiet noisy third-party loggers; opt in via --verbose.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
