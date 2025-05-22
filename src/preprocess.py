from pathlib import Path
import cv2

def resize_and_denoise(
    image_path: str | Path,
    output_path: str | Path,
    size: tuple[int, int] = (512, 512)
) -> None:
    """
    Perform geometric resizing and chromatic noise reduction on an input image.

    This function:
      1. Loads the source image from disk.
      2. Uniformly scales it to the prescribed dimensions using area-based interpolation.
      3. Applies non-local means color denoising to attenuate noise while preserving edges.
      4. Writes the processed image to the specified output path, creating parent directories if necessary.

    Args:
        image_path (str | Path):
            Filesystem path to the original image.
        output_path (str | Path):
            Destination path for the processed image. Parent directories will be created as needed.
        size (tuple[int, int], optional):
            Target width and height, in pixels. Defaults to (512, 512).

    Raises:
        FileNotFoundError:
            If the source image cannot be loaded.
        IOError:
            If writing the processed image to disk fails.
    """
    image_path = Path(image_path)
    output_path = Path(output_path)

    # Load the input image; abort if unavailable.
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    # Rescale the image to the desired spatial resolution.
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    # Reduce color and luminance noise using the non-local means algorithm.
    denoised = cv2.fastNlMeansDenoisingColored(resized, None, 10, 10, 7, 21)

    # Guarantee that the output directory structure exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Persist the denoised image; raise an exception on failure.
    if not cv2.imwrite(str(output_path), denoised):
        raise IOError(f"Failed to write preprocessed image to: {output_path}")
