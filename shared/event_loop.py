"""Event loop helpers with optional uvloop acceleration.

Provides a single function to create an asyncio event loop with uvloop
if available, falling back to the standard asyncio loop otherwise.

uvloop is a drop-in replacement for asyncio's event loop that provides
2-4x throughput improvement on Linux and macOS. It is not available on
Windows.

Usage::

    from shared.event_loop import create_event_loop
    loop = create_event_loop()
    asyncio.set_event_loop(loop)
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

_UVLOOP_AVAILABLE: bool = False
_UVLOOP_CHECKED: bool = False


def _check_uvloop() -> bool:
    """Check if uvloop is importable. Result is cached."""
    global _UVLOOP_AVAILABLE, _UVLOOP_CHECKED
    if not _UVLOOP_CHECKED:
        try:
            import uvloop  # noqa: F401
            _UVLOOP_AVAILABLE = True
        except ImportError:
            _UVLOOP_AVAILABLE = False
        _UVLOOP_CHECKED = True
    return _UVLOOP_AVAILABLE


def create_event_loop() -> asyncio.AbstractEventLoop:
    """Create a new event loop, using uvloop if available.

    Returns a uvloop-backed event loop on Linux/macOS when the
    ``uvloop`` package is installed, otherwise a standard asyncio loop.
    """
    if _check_uvloop():
        import uvloop
        loop = uvloop.new_event_loop()
        logger.debug("Created uvloop event loop")
        return loop
    return asyncio.new_event_loop()


def install_uvloop_policy() -> bool:
    """Install uvloop as the default event loop policy.

    Call once at program startup (before any event loops are created)
    to make all future ``asyncio.new_event_loop()`` calls use uvloop.

    Returns True if uvloop was installed, False if unavailable.
    """
    if _check_uvloop():
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.info("Installed uvloop event loop policy")
        return True
    logger.debug("uvloop not available, using standard asyncio")
    return False
