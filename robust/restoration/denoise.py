import numpy as np
import cv2
from PIL import Image


def denoise_nlmeans(img: Image.Image, h: float = 10,
                    template_window: int = 7, search_window: int = 21) -> Image.Image:
    """OpenCV Non-Local Means denoising."""
    arr = np.array(img.convert("RGB"))
    denoised = cv2.fastNlMeansDenoisingColored(arr, None, h, h, template_window, search_window)
    return Image.fromarray(denoised)


def denoise_gaussian(img: Image.Image, sigma: float = 1.5) -> Image.Image:
    """Fast Gaussian smoothing."""
    arr = np.array(img)
    k = int(6 * sigma + 1) | 1
    return Image.fromarray(cv2.GaussianBlur(arr, (k, k), sigma))
