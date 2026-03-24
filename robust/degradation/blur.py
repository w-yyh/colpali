import numpy as np
import cv2
from PIL import Image


def add_gaussian_blur(img: Image.Image, sigma: float = 3) -> Image.Image:
    arr = np.array(img)
    k = int(6 * sigma + 1) | 1
    return Image.fromarray(cv2.GaussianBlur(arr, (k, k), sigma))


def add_motion_blur(img: Image.Image, kernel_size: int = 15, angle: float = 45) -> Image.Image:
    arr = np.array(img)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    M = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    return Image.fromarray(cv2.filter2D(arr, -1, kernel))
