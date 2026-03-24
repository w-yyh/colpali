# tests/test_degradation.py
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import numpy as np
from PIL import Image
import pytest

@pytest.fixture
def sample_image():
    arr = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    return Image.fromarray(arr.astype(np.uint8))

def test_gaussian_noise_changes_image(sample_image):
    from robust.degradation.noise import add_gaussian_noise
    noisy = add_gaussian_noise(sample_image, std=25)
    assert not np.array_equal(np.array(sample_image), np.array(noisy))

def test_gaussian_noise_bounded(sample_image):
    from robust.degradation.noise import add_gaussian_noise
    arr = np.array(add_gaussian_noise(sample_image, std=50))
    assert arr.min() >= 0 and arr.max() <= 255

def test_motion_blur(sample_image):
    from robust.degradation.blur import add_motion_blur
    blurred = add_motion_blur(sample_image, kernel_size=15, angle=45)
    assert not np.array_equal(np.array(sample_image), np.array(blurred))

def test_jpeg_compression(sample_image):
    from robust.degradation.jpeg import add_jpeg_compression
    compressed = add_jpeg_compression(sample_image, quality=5)
    assert not np.array_equal(np.array(sample_image), np.array(compressed))

def test_tilt(sample_image):
    from robust.degradation.tilt import add_tilt
    tilted = add_tilt(sample_image, angle=15)
    assert isinstance(tilted, Image.Image)
    assert tilted.size == sample_image.size

def test_pipeline_composition(sample_image):
    from robust.degradation.pipeline import DegradationPipeline
    pipeline = DegradationPipeline([
        ("gaussian_noise", {"std": 20}),
        ("motion_blur", {"kernel_size": 9, "angle": 30}),
    ])
    result = pipeline(sample_image)
    assert isinstance(result, Image.Image)
    assert not np.array_equal(np.array(sample_image), np.array(result))

def test_pipeline_unknown_type(sample_image):
    from robust.degradation.pipeline import DegradationPipeline
    with pytest.raises(ValueError, match="Unknown degradation"):
        DegradationPipeline([("nonexistent", {})])
