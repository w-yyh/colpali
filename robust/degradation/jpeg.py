import io
from PIL import Image


def add_jpeg_compression(img: Image.Image, quality: int = 10) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()
