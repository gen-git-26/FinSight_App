# tools/async_utils.py
import asyncio
import threading
from typing import TypeVar, Callable, Any
from functools import wraps

T = TypeVar('T')

def run_async_in_thread(coro):
    """
    Run an async function in a separate thread to avoid event loop conflicts.
    Safe to call from Streamlit or other async contexts.
    """
    def run_in_new_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    thread = threading.Thread(target=run_in_new_loop, daemon=False)
    thread.start()
    thread.join()  # Wait for completion
    
    # Get result (simplified - for production use queue)
    return None  # Placeholder


def sync_to_async_wrapper(func: Callable) -> Callable:
    """
    Decorator to safely call async functions from sync context (like Streamlit).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Check if there's already a running loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - safe to use asyncio.run()
            return asyncio.run(func(*args, **kwargs))
        
        # There's a running loop - run in thread pool
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, func(*args, **kwargs))
            return future.result()
    
    return wrapper