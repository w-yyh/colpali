from PIL import Image, ImageDraw


def add_text_watermark(img: Image.Image, text: str = "WATERMARK", alpha: float = 0.3) -> Image.Image:
    img_rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img_rgba.size
    draw.text((w // 4, h // 2), text, fill=(255, 0, 0, int(255 * alpha)))
    return Image.alpha_composite(img_rgba, overlay).convert("RGB")
