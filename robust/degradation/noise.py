import numpy as np
from PIL import Image


def add_gaussian_noise(img: Image.Image, std: float = 25) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def add_salt_pepper_noise(img: Image.Image, amount: float = 0.05) -> Image.Image:
    arr = np.array(img).copy()
    n = int(amount * arr.size)
    for val in [255, 0]:
        coords = [np.random.randint(0, s, n) for s in arr.shape[:2]]
        arr[coords[0], coords[1]] = val
    return Image.fromarray(arr)
