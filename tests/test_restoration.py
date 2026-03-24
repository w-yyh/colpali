# tests/test_restoration.py
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import numpy as np
from PIL import Image
import pytest

@pytest.fixture
def noisy_image():
    base = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
    noise = np.random.normal(0, 40, base.shape).astype(np.int16)
    return Image.fromarray(np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8))

def test_nlmeans_returns_pil(noisy_image):
    from robust.restoration.denoise import denoise_nlmeans
    result = denoise_nlmeans(noisy_image)
    assert isinstance(result, Image.Image)
    assert result.size == noisy_image.size

def test_nlmeans_reduces_noise(noisy_image):
    from robust.restoration.denoise import denoise_nlmeans
    denoised = denoise_nlmeans(noisy_image)
    orig_std = np.array(noisy_image).astype(float).std()
    denoised_std = np.array(denoised).astype(float).std()
    assert denoised_std < orig_std * 0.9, (
        f"Expected denoised_std < {orig_std*0.9:.1f}, got {denoised_std:.1f}"
    )

def test_restoration_pipeline(noisy_image):
    from robust.restoration.pipeline import RestorationPipeline
    pipeline = RestorationPipeline(steps=[("nlmeans", {})])
    result = pipeline(noisy_image)
    assert isinstance(result, Image.Image)

def test_pipeline_unknown():
    from robust.restoration.pipeline import RestorationPipeline
    with pytest.raises(ValueError, match="Unknown restoration"):
        RestorationPipeline([("bogus", {})])
