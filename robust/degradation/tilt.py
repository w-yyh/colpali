from PIL import Image


def add_tilt(img: Image.Image, angle: float = 10) -> Image.Image:
    """Simulate a scanned document with slight rotation/skew."""
    # fillcolor must match the image mode: single int for grayscale, tuple for RGB/RGBA
    if img.mode in ("L", "P", "1"):
        fillcolor = 255
    else:
        fillcolor = (255, 255, 255)
    return img.rotate(angle, expand=False, fillcolor=fillcolor)
