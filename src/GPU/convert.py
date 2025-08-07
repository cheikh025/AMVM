import os
from PIL import Image
import numpy as np

VALID_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
def process_images(input_folder: str,
                   output_folder: str,
                   new_size: tuple[int,int],
                   transform_fn,
                   valid_exts: tuple[str,...] = VALID_EXTS):
    """
    Walk `input_folder`, resize+grayscale each image to `new_size`,
    apply `transform_fn(arr: np.ndarray) -> np.ndarray`, then save into `output_folder`.
    """
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(valid_exts):
            continue

        in_path  = os.path.join(input_folder,  fname)
        out_path = os.path.join(output_folder, fname)

        with Image.open(in_path) as img:
            img_resized = img.resize(new_size, resample=Image.Resampling.LANCZOS)
            img_gray    = img_resized.convert('L')
            arr         = np.array(img_gray, dtype=np.uint8)
            arr_out     = transform_fn(arr)
            Image.fromarray(arr_out).save(out_path)
            print(f"Processed and saved: {out_path}")


def binary_threshold(arr: np.ndarray, threshold: int = 50) -> np.ndarray:
    """Map pixels < threshold to 0, else to 255."""
    return np.where(arr < threshold, 0, 255).astype(np.uint8)

def quantize_nbit(arr: np.ndarray, bits: int) -> np.ndarray:
    """
    Uniformly quantize to `2**bits` levels across [0,255].
    e.g. bits=3 → 8 gray‐levels; bits=4 → 16 levels.
    """
    levels = 2 ** bits
    scaled    = np.round(arr / 255 * (levels - 1))
    recon     = (scaled / (levels - 1) * 255)
    return recon.astype(np.uint8)

def quantize_2bit(arr: np.ndarray) -> np.ndarray:
    return quantize_nbit(arr, bits=2)

def quantize_3bit(arr: np.ndarray) -> np.ndarray:
    return quantize_nbit(arr, bits=3)

def quantize_4bit(arr: np.ndarray) -> np.ndarray:
    return quantize_nbit(arr, bits=4)



if __name__ == "__main__":
    src = './br/'
    size = (128, 128)

    # 1) Binary 
    process_images(
        input_folder=src,
        output_folder='./data1/1bit/',
        new_size=size,
        transform_fn=lambda arr: binary_threshold(arr, threshold=127)
    )

    # 2) 4-bit quantization
    process_images(
        input_folder=src,
        output_folder='./data1/4bit/',
        new_size=size,
        transform_fn=quantize_4bit
    )

    # 3) 3-bit quantization
    process_images(
        input_folder=src,
        output_folder='./data1/3bit/',
        new_size=size,
        transform_fn=quantize_3bit
    )
    # 2) 2-bit quantization
    process_images(
        input_folder=src,
        output_folder='./data1/2bit/',
        new_size=size,
        transform_fn=quantize_2bit
    )
