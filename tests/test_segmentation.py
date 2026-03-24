# tests/test_segmentation.py
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import numpy as np
from PIL import Image
import pytest

@pytest.fixture
def document_image():
    """White background with dark document area in center."""
    arr = np.ones((400, 300, 3), dtype=np.uint8) * 240
    arr[50:350, 30:270] = 30
    return Image.fromarray(arr)

def test_segment_returns_pil(document_image):
    from robust.segmentation.document_seg import segment_document
    assert isinstance(segment_document(document_image), Image.Image)

def test_segment_same_size(document_image):
    from robust.segmentation.document_seg import segment_document
    result = segment_document(document_image)
    assert result.size == document_image.size

def test_segment_background_white(document_image):
    from robust.segmentation.document_seg import segment_document
    result = segment_document(document_image)
    arr = np.array(result)
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    for corner in corners:
        assert corner.mean() > 200, f"Corner {corner} should be white"
