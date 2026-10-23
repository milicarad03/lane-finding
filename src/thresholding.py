"""Create a binary lane-marking mask from an undistorted road image.

Run this module from the project root:

    python src/thresholding.py
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from .undistort import load_calibration, scale_camera_matrix, undistort_image
except ImportError:
    from undistort import load_calibration, scale_camera_matrix, undistort_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = PROJECT_ROOT / "calibration_data" / "calibration.npz"
DEFAULT_INPUT_IMAGE = PROJECT_ROOT / "test_images" / "test1.jpg"
OUTPUT_DIR = PROJECT_ROOT / "output_images" / "thresholding"


@dataclass(frozen=True)
class ThresholdConfig:
    """HLS and Sobel thresholds used to identify lane markings."""

    white_lightness: tuple[int, int] = (200, 255)
    white_saturation: tuple[int, int] = (0, 80)
    yellow_hue: tuple[int, int] = (15, 35)
    yellow_lightness: tuple[int, int] = (30, 255)
    yellow_saturation: tuple[int, int] = (80, 255)
    sobel_x: tuple[int, int] = (20, 100)
    sobel_kernel: int = 3
    gradient_lightness_min: int = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create color and gradient lane-marking masks."
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        default=DEFAULT_INPUT_IMAGE,
        help="road image to process (default: test_images/test1.jpg)",
    )
    return parser.parse_args()


def validate_image(image: np.ndarray) -> None:
    if image is None or image.size == 0:
        raise RuntimeError("Cannot process an empty image.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError("Expected a three-channel BGR image.")


def white_lane_mask(
    lightness: np.ndarray,
    saturation: np.ndarray,
    lightness_threshold: tuple[int, int] = (200, 255),
    saturation_threshold: tuple[int, int] = (0, 80),
) -> np.ndarray:
    """Return a 0/1 mask for bright, low-saturation white markings."""

    selected = (
        (lightness >= lightness_threshold[0])
        & (lightness <= lightness_threshold[1])
        & (saturation >= saturation_threshold[0])
        & (saturation <= saturation_threshold[1])
    )
    return selected.astype(np.uint8)


def yellow_lane_mask(
    hue: np.ndarray,
    lightness: np.ndarray,
    saturation: np.ndarray,
    hue_threshold: tuple[int, int] = (15, 35),
    lightness_threshold: tuple[int, int] = (30, 255),
    saturation_threshold: tuple[int, int] = (80, 255),
) -> np.ndarray:
    """Return a 0/1 mask for saturated yellow markings."""

    selected = (
        (hue >= hue_threshold[0])
        & (hue <= hue_threshold[1])
        & (lightness >= lightness_threshold[0])
        & (lightness <= lightness_threshold[1])
        & (saturation >= saturation_threshold[0])
        & (saturation <= saturation_threshold[1])
    )
    return selected.astype(np.uint8)


def sobel_x_mask(
    lightness: np.ndarray,
    threshold: tuple[int, int] = (20, 100),
    kernel_size: int = 3,
) -> np.ndarray:
    """Return a 0/1 mask for horizontal changes in lightness."""

    gradient = cv2.Sobel(
        lightness,
        cv2.CV_64F,
        dx=1,
        dy=0,
        ksize=kernel_size,
    )
    absolute_gradient = np.absolute(gradient)
    maximum = float(absolute_gradient.max())

    if maximum == 0.0:
        return np.zeros_like(lightness, dtype=np.uint8)

    scaled = np.uint8(np.clip(255.0 * absolute_gradient / maximum, 0, 255))
    selected = (scaled >= threshold[0]) & (scaled <= threshold[1])
    return selected.astype(np.uint8)


def create_threshold_masks(
    image: np.ndarray,
    config: ThresholdConfig = ThresholdConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the white, yellow, Sobel, and combined binary masks."""

    validate_image(image)
    hue, lightness, saturation = cv2.split(
        cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    )

    white = white_lane_mask(
        lightness,
        saturation,
        config.white_lightness,
        config.white_saturation,
    )
    yellow = yellow_lane_mask(
        hue,
        lightness,
        saturation,
        config.yellow_hue,
        config.yellow_lightness,
        config.yellow_saturation,
    )
    sobel = sobel_x_mask(lightness, config.sobel_x, config.sobel_kernel)

    bright_gradient = (sobel == 1) & (
        lightness >= config.gradient_lightness_min
    )
    combined = ((white == 1) | (yellow == 1) | bright_gradient).astype(np.uint8)
    return white, yellow, sobel, combined


def combined_threshold(
    image: np.ndarray,
    config: ThresholdConfig = ThresholdConfig(),
) -> np.ndarray:
    """Return the final 0/1 mask for an already-undistorted BGR image."""

    return create_threshold_masks(image, config)[3]


def mask_for_saving(mask: np.ndarray) -> np.ndarray:
    return (mask * 255).astype(np.uint8)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    labelled = image.copy()
    cv2.putText(
        labelled,
        label,
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return labelled


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def main() -> int:
    args = parse_args()
    input_path = (
        args.input_image
        if args.input_image.is_absolute()
        else PROJECT_ROOT / args.input_image
    )

    try:
        calibration = load_calibration(CALIBRATION_FILE)
        if not input_path.is_file():
            raise RuntimeError(f"Input image not found: {input_path}")

        original = cv2.imread(str(input_path))
        if original is None:
            raise RuntimeError(
                f"OpenCV could not decode the input image: {input_path}"
            )

        input_size = (original.shape[1], original.shape[0])
        camera_matrix = scale_camera_matrix(
            calibration.camera_matrix,
            calibration.image_size,
            input_size,
        )

        undistorted = undistort_image(
            original,
            camera_matrix,
            calibration.distortion_coefficients,
        )
        white, yellow, sobel, combined = create_threshold_masks(undistorted)

        output_paths = {
            "white": OUTPUT_DIR / f"{input_path.stem}_white_mask.png",
            "yellow": OUTPUT_DIR / f"{input_path.stem}_yellow_mask.png",
            "sobel": OUTPUT_DIR / f"{input_path.stem}_sobel_mask.png",
            "binary": OUTPUT_DIR / f"{input_path.stem}_binary.png",
            "comparison": OUTPUT_DIR / f"{input_path.stem}_comparison.jpg",
        }
        write_image(output_paths["white"], mask_for_saving(white))
        write_image(output_paths["yellow"], mask_for_saving(yellow))
        write_image(output_paths["sobel"], mask_for_saving(sobel))
        binary_image = mask_for_saving(combined)
        write_image(output_paths["binary"], binary_image)

        comparison = cv2.hconcat(
            [
                add_label(original, "Original"),
                add_label(undistorted, "Undistorted"),
                add_label(
                    cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR),
                    "Combined binary",
                ),
            ]
        )
        write_image(output_paths["comparison"], comparison)
    except (RuntimeError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Input image: {input_path}")
    print(f"Output directory: {OUTPUT_DIR}")
    for output_path in output_paths.values():
        print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
