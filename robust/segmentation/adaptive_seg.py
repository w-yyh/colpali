"""
Improved document segmentation using adaptive thresholding + connected component analysis.

Key improvements over baseline Otsu method:
  1. Adaptive threshold instead of global Otsu → handles uneven lighting
  2. Connected component analysis → filters small noise, keeps major content regions
  3. Support both 'whiten' and 'crop' modes
  4. Tunable parameters for PSO optimization
"""
import numpy as np
import cv2
from PIL import Image


def adaptive_segment(
    img: Image.Image,
    block_size: int = 51,
    c_offset: int = 10,
    morph_kernel_size: int = 15,
    min_area_ratio: float = 0.001,
    padding: int = 10,
    mode: str = "whiten",
) -> Image.Image:
    """
    Segment document foreground using adaptive thresholding and connected components.

    Parameters
    ----------
    img : PIL.Image
        Input document image (RGB).
    block_size : int
        Size of the neighborhood for adaptive threshold (must be odd, >= 3).
    c_offset : int
        Constant subtracted from the mean in adaptive threshold.
    morph_kernel_size : int
        Size of the morphological closing kernel to merge nearby regions.
    min_area_ratio : float
        Minimum connected component area as a fraction of total image area.
        Components smaller than this are discarded as noise.
    padding : int
        Padding (pixels) around the detected bounding box.
    mode : str
        'whiten' — set background to white, keep original image size.
        'crop'   — crop to the bounding box of foreground regions.

    Returns
    -------
    PIL.Image
        Processed document image.
    """
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    total_area = h * w

    # Ensure block_size is odd and >= 3
    block_size = max(3, block_size)
    if block_size % 2 == 0:
        block_size += 1

    # Adaptive thresholding (inverted: foreground = 255)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_offset
    )

    # Morphological closing to merge nearby text/content regions
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Connected component analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    # Filter components: keep those larger than min_area_ratio * total_area
    min_area = total_area * min_area_ratio
    mask = np.zeros(gray.shape, dtype=np.uint8)

    for i in range(1, num_labels):  # skip label 0 (background)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            mask[labels == i] = 255

    # If nothing survived the filter, return original
    if mask.sum() == 0:
        return img

    # Find bounding box of all retained components
    coords = cv2.findNonZero(mask)
    x, y, bw, bh = cv2.boundingRect(coords)

    # Apply padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    bw = min(w - x, bw + 2 * padding)
    bh = min(h - y, bh + 2 * padding)

    if mode == "crop":
        return Image.fromarray(arr[y:y + bh, x:x + bw])

    # Default: whiten mode — keep size, set background to white
    result = arr.copy()
    box_mask = np.zeros(gray.shape, dtype=np.uint8)
    box_mask[y:y + bh, x:x + bw] = 255
    result[box_mask == 0] = 255
    return Image.fromarray(result)
