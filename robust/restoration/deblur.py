import numpy as np
from PIL import Image


def _wiener_numpy(arr: np.ndarray, size: tuple, noise_power: float) -> np.ndarray:
    """Pure-numpy Wiener filter fallback (frequency-domain)."""
    from numpy.fft import fft2, ifft2
    F = fft2(arr)
    power = np.abs(F) ** 2
    H = power / (power + noise_power * arr.size)
    restored = np.real(ifft2(H * F))
    return restored


def deblur_wiener(img: Image.Image, noise_power: float = 0.01) -> Image.Image:
    """Wiener filter deblur (frequency domain). Uses scipy if available, else numpy fallback."""
    arr = np.array(img.convert("L")).astype(np.float32) / 255.0
    try:
        from scipy.signal import wiener as scipy_wiener  # scipy is a core colpali dep
        restored = scipy_wiener(arr, (5, 5), noise_power)
    except (ImportError, AttributeError):
        restored = _wiener_numpy(arr, (5, 5), noise_power)
    restored = np.clip(restored * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(restored).convert(img.mode)
