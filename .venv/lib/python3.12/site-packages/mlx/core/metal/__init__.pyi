

from mlx.core import array, Dtype, Device, Stream
from typing import Sequence, Optional, Union

def clear_cache() -> None:
    """
    Clear the memory cache.

    After calling this, :func:`get_cache_memory` should return ``0``.
    """

def device_info() -> dict[str, str | int]:
    """
    Get information about the GPU device and system settings.

    Currently returns:

    * ``architecture``
    * ``max_buffer_size``
    * ``max_recommended_working_set_size``
    * ``memory_size``

    Returns:
        dict: A dictionary with string keys and string or integer values.
    """

def get_active_memory() -> int:
    """
    Get the actively used memory in bytes.

    Note, this will not always match memory use reported by the system because
    it does not include cached memory buffers.
    """

def get_cache_memory() -> int:
    """
    Get the cache size in bytes.

    The cache includes memory not currently used that has not been returned
    to the system allocator.
    """

def get_peak_memory() -> int:
    """
    Get the peak amount of used memory in bytes.

    The maximum memory used recorded from the beginning of the program
    execution or since the last call to :func:`reset_peak_memory`.
    """

def is_available() -> bool:
    """Check if the Metal back-end is available."""

def reset_peak_memory() -> None:
    """Reset the peak memory to zero."""

def set_cache_limit(limit: int) -> int:
    """
    Set the free cache limit.

    If using more than the given limit, free memory will be reclaimed
    from the cache on the next allocation. To disable the cache, set
    the limit to ``0``.

    The cache limit defaults to the memory limit. See
    :func:`set_memory_limit` for more details.

    Args:
      limit (int): The cache limit in bytes.

    Returns:
      int: The previous cache limit in bytes.
    """

def set_memory_limit(limit: int, *, relaxed: bool = True) -> int:
    """
    Set the memory limit.

    Memory allocations will wait on scheduled tasks to complete if the limit
    is exceeded. If there are no more scheduled tasks an error will be raised
    if ``relaxed`` is ``False``. Otherwise memory will be allocated
    (including the potential for swap) if ``relaxed`` is ``True``.

    The memory limit defaults to 1.5 times the maximum recommended working set
    size reported by the device.

    Args:
      limit (int): Memory limit in bytes.
      relaxed (bool, optional): If `False`` an error is raised if the limit
        is exceeded. Default: ``True``

    Returns:
      int: The previous memory limit in bytes.
    """

def set_wired_limit(limit: int) -> int:
    """
    Set the wired size limit.

    .. note::
       * This function is only useful on macOS 15.0 or higher.
       * The wired limit should remain strictly less than the total
         memory size.

    The wired limit is the total size in bytes of memory that will be kept
    resident. The default value is ``0``.

    Setting a wired limit larger than system wired limit is an error. You can
    increase the system wired limit with:

    .. code-block::

      sudo sysctl iogpu.wired_limit_mb=<size_in_megabytes>

    Use :func:`device_info` to query the system wired limit
    (``"max_recommended_working_set_size"``) and the total memory size
    (``"memory_size"``).

    Args:
      limit (int): The wired limit in bytes.

    Returns:
      int: The previous wired limit in bytes.
    """

def start_capture(path: str) -> None:
    """
    Start a Metal capture.

    Args:
      path (str): The path to save the capture which should have
        the extension ``.gputrace``.
    """

def stop_capture() -> None:
    """Stop a Metal capture."""
