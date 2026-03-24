"""
Document body segmentation using Otsu thresholding + morphological close + contour detection.
Sets background pixels to white, reducing redundant patches fed to ColQwen2.
"""
import numpy as np
import cv2
from PIL import Image


def segment_document(img: Image.Image, padding: int = 5) -> Image.Image:
    """Find largest document contour, set background to white."""
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(arr.shape[1] - x, w + 2 * padding)
        h = min(arr.shape[0] - y, h + 2 * padding)
        mask[y:y+h, x:x+w] = 255
    else:
        mask[:] = 255
    result = arr.copy()
    result[mask == 0] = 255
    return Image.fromarray(result)
