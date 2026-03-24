from PIL import Image
from typing import List, Tuple, Dict, Any
from .noise import add_gaussian_noise, add_salt_pepper_noise
from .blur import add_gaussian_blur, add_motion_blur
from .tilt import add_tilt
from .jpeg import add_jpeg_compression
from .watermark import add_text_watermark

_REGISTRY = {
    "gaussian_noise":    add_gaussian_noise,
    "salt_pepper_noise": add_salt_pepper_noise,
    "gaussian_blur":     add_gaussian_blur,
    "motion_blur":       add_motion_blur,
    "tilt":              add_tilt,
    "jpeg_compression":  add_jpeg_compression,
    "watermark":         add_text_watermark,
}


class DegradationPipeline:
    def __init__(self, steps: List[Tuple[str, Dict[str, Any]]]):
        for name, _ in steps:
            if name not in _REGISTRY:
                raise ValueError(f"Unknown degradation: {name!r}. Available: {sorted(_REGISTRY)}")
        self.steps = steps

    def __call__(self, img: Image.Image) -> Image.Image:
        for name, params in self.steps:
            img = _REGISTRY[name](img, **params)
        return img

    @classmethod
    def available(cls):
        return sorted(_REGISTRY.keys())
