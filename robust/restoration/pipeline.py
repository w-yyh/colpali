from PIL import Image
from typing import List, Tuple, Dict, Any
from .denoise import denoise_nlmeans, denoise_gaussian
from .deblur import deblur_wiener

_REGISTRY = {
    "nlmeans":  denoise_nlmeans,
    "gaussian": denoise_gaussian,
    "wiener":   deblur_wiener,
}


class RestorationPipeline:
    def __init__(self, steps: List[Tuple[str, Dict[str, Any]]]):
        for name, _ in steps:
            if name not in _REGISTRY:
                raise ValueError(f"Unknown restoration: {name!r}. Available: {sorted(_REGISTRY)}")
        self.steps = steps

    def __call__(self, img: Image.Image) -> Image.Image:
        for name, params in self.steps:
            img = _REGISTRY[name](img, **params)
        return img
