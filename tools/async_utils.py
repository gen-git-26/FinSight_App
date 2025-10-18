# tools/async_utils.py
"""
Utilities for safely running async code from sync contexts (Streamlit, etc).
Handles the "asyncio.run() cannot be called from a running event loop" error.
"""

import asyncio
import threading
from typing import TypeVar, Callable, Any, Coroutine
from functools import wraps

T = TypeVar('T')


def run_async_safe(coro: Coroutine) -> Any:
    """
    Run an async coroutine safely from any context.
    
    If there's a running event loop (e.g., in Streamlit), runs in a thread.
    Otherwise, uses asyncio.run() directly.
    """
    try:
        # Check if we're already in an async context
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - safe to use asyncio.run()
        return asyncio.run(coro)
    
    # We're in an async context - run in a thread pool
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


def sync_callable(async_func: Callable) -> Callable:
    """
    Decorator to convert an async function to a safe sync callable.
    
    Usage:
        @sync_callable
        async def my_async_func(x: str) -> str:
            return x
        
        result = my_async_func("hello")  # Works from Streamlit!
    """
    @wraps(async_func)
    def wrapper(*args, **kwargs):
        coro = async_func(*args, **kwargs)
        return run_async_safe(coro)
    
    return wrapper


def run_in_new_event_loop(coro: Coroutine) -> Any:
    """
    Run coroutine in a completely new event loop (thread-safe).
    Useful when you want isolation from parent loop.
    """
    result = {}
    error = {}
    
    def run():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result['value'] = loop.run_until_complete(coro)
        except Exception as e:
            error['value'] = e
        finally:
            loop.close()
    
    thread = threading.Thread(target=run, daemon=False)
    thread.start()
    thread.join()
    
    if error:
        raise error['value']
    return result.get('value')