from collections.abc import Sequence

import mlx.core


def fft(a: mlx.core.array, n: int | None = None, axis: int = -1, stream: mlx.core.Stream | mlx.core.Device | None = None) -> mlx.core.array:
    """
    One dimensional discrete Fourier Transform.

    Args:
        a (array): The input array.
        n (int, optional): Size of the transformed axis. The
           corresponding axis in the input is truncated or padded with
           zeros to match ``n``. The default value is ``a.shape[axis]``.
        axis (int, optional): Axis along which to perform the FFT. The
           default is ``-1``.

    Returns:
        array: The DFT of the input along the given axis.
    """

def fft2(a: mlx.core.array, s: Sequence[int] | None = None, axes: Sequence[int] | None = [-2, -1], stream: mlx.core.Stream | mlx.core.Device | None = None) -> mlx.core.array:
    """
    Two dimensional discrete Fourier Transform.

    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes. The
           corresponding axes in the input are truncated or padded with
           zeros to match the sizes in ``s``. The default value is the
           sizes of ``a`` along ``axes``.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``[-2, -1]``.

    Returns:
        array: The DFT of the input along the given axes.
    """

def fftn(a: mlx.core.array, s: Sequence[int] | None = None, axes: Sequence[int] | None = None, stream: mlx.core.Stream | mlx.core.Device | None = None) -> mlx.core.array:
    """
    n-dimensional discrete Fourier Transform.

    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes. The
           corresponding axes in the input are truncated or padded with
           zeros to match the sizes in ``s``. The default value is the
           sizes of ``a`` along ``axes``.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``None`` in which case the FFT is over the last
           ``len(s)`` axes are or all axes if ``s`` is also ``None``.

    Returns:
        array: The DFT of the input along the given axes.
    """

def ifft(a: mlx.core.array, n: int | None = None, axis: int = -1, stream: mlx.core.Stream | mlx.core.Device | None = None) -> mlx.core.array:
    """
    One dimensional inverse discrete Fourier Transform.

    Args:
        a (array): The input array.
        n (int, optional): Size of the transformed axis. The
           corresponding axis in the input is truncated or padded with
           zeros to match ``n``. The default value is ``a.shape[axis]``.
        axis (int, optional): Axis along which to perform the FFT. The
           default is ``-1``.

    Returns:
        array: The inverse DFT of the input along the given axis.
    """

def ifft2(a: mlx.core.array, s: Sequence[int] | None = None, axes: Sequence[int] | None = [-2, -1], stream: mlx.core.Stream | mlx.core.Device | None = None) -> mlx.core.array:
    """
    Two dimensional inverse discrete Fourier Transform.

    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes. The
           corresponding axes in the input are truncated or padded with
           zeros to match the sizes in ``s``. The default value is the
           sizes of ``a`` along ``axes``.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``[-2, -1]``.

    Returns:
        array: The inverse DFT of the input along the given axes.
    """

def ifftn(a: mlx.core.array, s: Sequence[int] | None = None, axes: Sequence[int] | None = None, stream: mlx.core.Stream | mlx.core.Device | None = None) -> mlx.core.array:
    """
    n-dimensional inverse discrete Fourier Transform.

    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes. The
           corresponding axes in the input are truncated or padded with
           zeros to match the sizes in ``s``. The default value is the
           sizes of ``a`` along ``axes``.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``None`` in which case the FFT is over the last
           ``len(s)`` axes or all axes if ``s`` is also ``None``.

    Returns:
        array: The inverse DFT of the input along the given axes.
    """

def irfft(a: mlx.core.array, n: int | None = None, axis: int = -1, stream: mlx.core.Stream | mlx.core.Device | None = None) -> mlx.core.array:
    """
    The inverse of :func:`rfft`.

    The output has the same shape as the input except along ``axis`` in
    which case it has size ``n``.

    Args:
        a (array): The input array.
        n (int, optional): Size of the transformed axis. The
           corresponding axis in the input is truncated or padded with
           zeros to match ``n // 2 + 1``. The default value is
           ``a.shape[axis] // 2 + 1``.
        axis (int, optional): Axis along which to perform the FFT. The
           default is ``-1``.

    Returns:
        array: The real array containing the inverse of :func:`rfft`.
    """

def irfft2(a: mlx.core.array, s: Sequence[int] | None = None, axes: Sequence[int] | None = [-2, -1], stream: mlx.core.Stream | mlx.core.Device | None = None) -> mlx.core.array:
    """
    The inverse of :func:`rfft2`.

    Note the input is generally complex. The dimensions of the input
    specified in ``axes`` are padded or truncated to match the sizes
    from ``s``. The last axis in ``axes`` is treated as the real axis
    and will have size ``s[-1] // 2 + 1``.

    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes. The
           corresponding axes in the input are truncated or padded with
           zeros to match the sizes in ``s`` except for the last axis
           which has size ``s[-1] // 2 + 1``. The default value is the
           sizes of ``a`` along ``axes``.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``[-2, -1]``.

    Returns:
        array: The real array containing the inverse of :func:`rfft2`.
    """

def irfftn(a: mlx.core.array, s: Sequence[int] | None = None, axes: Sequence[int] | None = None, stream: mlx.core.Stream | mlx.core.Device | None = None) -> mlx.core.array:
    """
    The inverse of :func:`rfftn`.

    Note the input is generally complex. The dimensions of the input
    specified in ``axes`` are padded or truncated to match the sizes
    from ``s``. The last axis in ``axes`` is treated as the real axis
    and will have size ``s[-1] // 2 + 1``.

    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes. The
           corresponding axes in the input are truncated or padded with
           zeros to match the sizes in ``s``. The default value is the
           sizes of ``a`` along ``axes``.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``None`` in which case the FFT is over the last
           ``len(s)`` axes or all axes if ``s`` is also ``None``.

    Returns:
        array: The real array containing the inverse of :func:`rfftn`.
    """

def rfft(a: mlx.core.array, n: int | None = None, axis: int = -1, stream: mlx.core.Stream | mlx.core.Device | None = None) -> mlx.core.array:
    """
    One dimensional discrete Fourier Transform on a real input.

    The output has the same shape as the input except along ``axis`` in
    which case it has size ``n // 2 + 1``.

    Args:
        a (array): The input array. If the array is complex it will be silently
           cast to a real type.
        n (int, optional): Size of the transformed axis. The
           corresponding axis in the input is truncated or padded with
           zeros to match ``n``. The default value is ``a.shape[axis]``.
        axis (int, optional): Axis along which to perform the FFT. The
           default is ``-1``.

    Returns:
        array: The DFT of the input along the given axis. The output
        data type will be complex.
    """

def rfft2(a: mlx.core.array, s: Sequence[int] | None = None, axes: Sequence[int] | None = [-2, -1], stream: mlx.core.Stream | mlx.core.Device | None = None) -> mlx.core.array:
    """
    Two dimensional real discrete Fourier Transform.

    The output has the same shape as the input except along the dimensions in
    ``axes`` in which case it has sizes from ``s``. The last axis in ``axes`` is
    treated as the real axis and will have size ``s[-1] // 2 + 1``.

    Args:
        a (array): The input array. If the array is complex it will be silently
           cast to a real type.
        s (list(int), optional): Sizes of the transformed axes. The
           corresponding axes in the input are truncated or padded with
           zeros to match the sizes in ``s``. The default value is the
           sizes of ``a`` along ``axes``.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``[-2, -1]``.

    Returns:
        array: The real DFT of the input along the given axes. The output
        data type will be complex.
    """

def rfftn(a: mlx.core.array, s: Sequence[int] | None = None, axes: Sequence[int] | None = None, stream: mlx.core.Stream | mlx.core.Device | None = None) -> mlx.core.array:
    """
    n-dimensional real discrete Fourier Transform.

    The output has the same shape as the input except along the dimensions in
    ``axes`` in which case it has sizes from ``s``. The last axis in ``axes`` is
    treated as the real axis and will have size ``s[-1] // 2 + 1``.

    Args:
        a (array): The input array. If the array is complex it will be silently
           cast to a real type.
        s (list(int), optional): Sizes of the transformed axes. The
           corresponding axes in the input are truncated or padded with
           zeros to match the sizes in ``s``. The default value is the
           sizes of ``a`` along ``axes``.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``None`` in which case the FFT is over the last
           ``len(s)`` axes or all axes if ``s`` is also ``None``.

    Returns:
        array: The real DFT of the input along the given axes. The output
    """
