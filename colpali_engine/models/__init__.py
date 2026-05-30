"""Lazy model exports.

The project only needs ColQwen2 for several local HR experiments. Importing all
model families eagerly makes those experiments sensitive to unrelated optional
Transformers modules, so public model names are resolved on first access.
"""

from importlib import import_module

_EXPORTS = {
    "BiGemma3": ".gemma3",
    "BiGemmaProcessor3": ".gemma3",
    "ColGemma3": ".gemma3",
    "ColGemmaProcessor3": ".gemma3",
    "BiIdefics3": ".idefics3",
    "BiIdefics3Processor": ".idefics3",
    "ColIdefics3": ".idefics3",
    "ColIdefics3Processor": ".idefics3",
    "BiModernVBert": ".modernvbert",
    "BiModernVBertProcessor": ".modernvbert",
    "ColModernVBert": ".modernvbert",
    "ColModernVBertProcessor": ".modernvbert",
    "BiPali": ".paligemma",
    "BiPaliProcessor": ".paligemma",
    "BiPaliProj": ".paligemma",
    "ColPali": ".paligemma",
    "ColPaliProcessor": ".paligemma",
    "BiQwen2": ".qwen2",
    "BiQwen2Processor": ".qwen2",
    "ColQwen2": ".qwen2",
    "ColQwen2Processor": ".qwen2",
    "BiQwen2_5": ".qwen2_5",
    "BiQwen2_5_Processor": ".qwen2_5",
    "ColQwen2_5": ".qwen2_5",
    "ColQwen2_5_Processor": ".qwen2_5",
    "BiQwen3": ".qwen3",
    "BiQwen3Processor": ".qwen3",
    "ColQwen3": ".qwen3",
    "ColQwen3Processor": ".qwen3",
    "BiQwen3_5": ".qwen3_5",
    "BiQwen3_5Processor": ".qwen3_5",
    "ColQwen3_5": ".qwen3_5",
    "ColQwen3_5Processor": ".qwen3_5",
    "ColQwen2_5Omni": ".qwen_omni",
    "ColQwen2_5OmniProcessor": ".qwen_omni",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
