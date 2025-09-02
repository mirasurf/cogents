import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Coroutine, ParamSpec, TypeVar

# Define generic type variables for return type and parameters
R = TypeVar("R")
T = TypeVar("T")
P = ParamSpec("P")


def _log_pretty_url(s: str, max_len: int | None = 22) -> str:
    """Truncate/pretty-print a URL with a maximum length, removing the protocol and www. prefix"""
    s = s.replace("https://", "").replace("http://", "").replace("www.", "")
    if max_len is not None and len(s) > max_len:
        return s[:max_len] + "…"
    return s


def _log_pretty_path(path: str | Path | None) -> str:
    """Pretty-print a path, shorten home dir to ~ and cwd to ."""

    if not path or not str(path).strip():
        return ""  # always falsy in -> falsy out so it can be used in ternaries

    # dont print anything thats not a path
    if not isinstance(path, (str, Path)):
        # no other types are safe to just str(path) and log to terminal unless we know what they are
        # e.g. what if we get storage_date=dict | Path and the dict version could contain real cookies
        return f"<{type(path).__name__}>"

    # replace home dir and cwd with ~ and .
    pretty_path = str(path).replace(str(Path.home()), "~").replace(str(Path.cwd().resolve()), ".")

    # wrap in quotes if it contains spaces
    if pretty_path.strip() and " " in pretty_path:
        pretty_path = f'"{pretty_path}"'

    return pretty_path


def is_new_tab_page(url: str) -> bool:
    """
    Check if a URL is a new tab page (about:blank, chrome://new-tab-page, or chrome://newtab).

    Args:
            url: The URL to check

    Returns:
            bool: True if the URL is a new tab page, False otherwise
    """
    return url in (
        "about:blank",
        "chrome://new-tab-page/",
        "chrome://new-tab-page",
        "chrome://newtab/",
        "chrome://newtab",
    )


def time_execution_async(
    additional_text: str = "",
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    def decorator(func: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            # Only log if execution takes more than 0.25 seconds to avoid spamming the logs
            # you can lower this threshold locally when you're doing dev work to performance optimize stuff
            if execution_time > 0.25:
                self_has_logger = args and getattr(args[0], "logger", None)
                if self_has_logger:
                    logger = getattr(args[0], "logger")
                elif "agent" in kwargs:
                    logger = getattr(kwargs["agent"], "logger")
                elif "browser_session" in kwargs:
                    logger = getattr(kwargs["browser_session"], "logger")
                else:
                    logger = logging.getLogger(__name__)
                logger.debug(f'⏳ {additional_text.strip("-")}() took {execution_time:.2f}s')
            return result

        return wrapper

    return decorator
