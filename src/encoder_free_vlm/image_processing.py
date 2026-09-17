from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps


def load_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    if isinstance(image, (str, Path)):
        image_str = str(image)
        if image_str.startswith(("http://", "https://")):
            import io
            import urllib.request
            req = urllib.request.Request(image_str, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                with Image.open(io.BytesIO(resp.read())) as opened:
                    return ImageOps.exif_transpose(opened).convert("RGB")
        with Image.open(image) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")
    if isinstance(image, dict) and "path" in image:
        return load_image(image["path"])
    if isinstance(image, bytes):
        import io

        with Image.open(io.BytesIO(image)) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def resize_shorter_side(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image has invalid dimensions")
    scale = size / min(width, height)
    new_width = max(size, round(width * scale))
    new_height = max(size, round(height * scale))
    resample = getattr(Image, "Resampling", Image).BICUBIC
    return image.resize((new_width, new_height), resample=resample)


def center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = max(0, (width - size) // 2)
    top = max(0, (height - size) // 2)
    return image.crop((left, top, left + size, top + size))


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected an RGB image")
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()
    return tensor.float().div_(255.0)


def preprocess_image(image: Any, image_size: int = 512) -> torch.Tensor:
    pil_image = load_image(image)
    pil_image = resize_shorter_side(pil_image, image_size)
    pil_image = center_crop(pil_image, image_size)
    return image_to_tensor(pil_image)
