

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_FILE = PROJECT_ROOT / "calibration_data" / "calibration.npz"
DEFAULT_INPUT_IMAGE = PROJECT_ROOT / "test_images" / "test1.jpg"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output_images" / "undistortion"


@dataclass(frozen=True)
class CalibrationData:
    """Intrinsic camera parameters required for distortion correction."""

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_size: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correct lens distortion in a road image."
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        default=DEFAULT_INPUT_IMAGE,
        help="road image to correct (default: test_images/test1.jpg)",
    )
    return parser.parse_args()


def load_calibration(path: Path) -> CalibrationData:
    """Load and validate the camera parameters produced by calibration.py."""

    if not path.is_file():
        raise RuntimeError(
            f"Calibration file not found: {path}. Run camera calibration first."
        )

    try:
        with np.load(path, allow_pickle=False) as saved:
            required_keys = {
                "camera_matrix",
                "distortion_coefficients",
                "image_size",
            }
            missing_keys = required_keys.difference(saved.files)
            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise RuntimeError(
                    f"Calibration file is missing required data: {missing}."
                )

            camera_matrix = np.asarray(saved["camera_matrix"], dtype=np.float64)
            distortion_coefficients = np.asarray(
                saved["distortion_coefficients"], dtype=np.float64
            )
            saved_image_size = np.asarray(saved["image_size"]).reshape(-1)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Could not load calibration data from {path}: {error}"
        ) from error

    if saved_image_size.size != 2:
        raise RuntimeError(
            "Invalid saved image size; expected width and height."
        )

    image_size = (int(saved_image_size[0]), int(saved_image_size[1]))
    return CalibrationData(
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        image_size=image_size,
    )


def scale_camera_matrix(
    camera_matrix: np.ndarray,
    calibration_size: tuple[int, int],
    input_size: tuple[int, int],
) -> np.ndarray:
    """Scale camera intrinsics for a proportionally resized image."""

    calibration_width, calibration_height = calibration_size
    input_width, input_height = input_size
    if min(calibration_width, calibration_height, input_width, input_height) <= 0:
        raise RuntimeError("Image dimensions must be positive.")

    calibration_ratio = calibration_width / calibration_height
    input_ratio = input_width / input_height
    if not np.isclose(input_ratio, calibration_ratio, rtol=1e-3):
        raise RuntimeError(
            f"Input aspect ratio {input_width}:{input_height} is incompatible "
            f"with calibration aspect ratio {calibration_width}:{calibration_height}."
        )

    if input_size == calibration_size:
        return camera_matrix.copy()

    scale_x = input_width / calibration_width
    scale_y = input_height / calibration_height
    scaled_matrix = camera_matrix.copy()
    scaled_matrix[0, 0] *= scale_x  # fx
    scaled_matrix[0, 2] *= scale_x  # cx
    scaled_matrix[1, 1] *= scale_y  # fy
    scaled_matrix[1, 2] *= scale_y  # cy

    print(
        "Warning: camera intrinsics were scaled for the input resolution. "
        "This assumes the image comes from the calibrated camera and was only resized.",
        file=sys.stderr,
    )
    return scaled_matrix


def undistort_image(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> np.ndarray:
    """Return a distortion-corrected copy of an image.

    This function is intentionally independent of file handling so later lane
    detection stages can reuse it for already-loaded images.
    """

    if image is None or image.size == 0:
        raise RuntimeError("Cannot undistort an empty image.")

    return cv2.undistort(image, camera_matrix, distortion_coefficients)


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
        calibration = load_calibration(DEFAULT_CALIBRATION_FILE)

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

        corrected = undistort_image(
            original,
            camera_matrix,
            calibration.distortion_coefficients,
        )

        undistorted_path = (
            DEFAULT_OUTPUT_DIR / f"{input_path.stem}_undistorted.jpg"
        )
        comparison_path = DEFAULT_OUTPUT_DIR / f"{input_path.stem}_comparison.jpg"
        comparison = cv2.hconcat(
            [add_label(original, "Original"), add_label(corrected, "Undistorted")]
        )

        write_image(undistorted_path, corrected)
        write_image(comparison_path, comparison)
    except (RuntimeError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Input image: {input_path}")
    print(f"Image resolution: {input_size[0]}x{input_size[1]}")
    print(f"Undistorted image saved to: {undistorted_path}")
    print(f"Side-by-side comparison saved to: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
