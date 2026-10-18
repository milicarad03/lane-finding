"""Correct lens distortion in a road image using saved camera parameters.

Run this module from the project root:

    python src/undistort.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_FILE = PROJECT_ROOT / "calibration_data" / "calibration.npz"
DEFAULT_INPUT_IMAGE = PROJECT_ROOT / "test_images" / "test1.jpg"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output_images" / "undistortion"


class DistortionCorrectionError(RuntimeError):
    """Raised when an image cannot be undistorted safely."""


@dataclass(frozen=True)
class CalibrationData:
    """Intrinsic camera parameters required for distortion correction."""

    camera_matrix: NDArray[np.float64]
    distortion_coefficients: NDArray[np.float64]
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
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=DEFAULT_CALIBRATION_FILE,
        help="saved camera parameters (default: calibration_data/calibration.npz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for generated images",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    """Resolve relative command-line paths from the project root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def load_calibration(path: Path) -> CalibrationData:
    """Load and validate the camera parameters produced by calibration.py."""

    if not path.is_file():
        raise DistortionCorrectionError(
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
                raise DistortionCorrectionError(
                    f"Calibration file is missing required data: {missing}."
                )

            camera_matrix = np.asarray(saved["camera_matrix"], dtype=np.float64)
            distortion_coefficients = np.asarray(
                saved["distortion_coefficients"], dtype=np.float64
            )
            saved_image_size = np.asarray(saved["image_size"]).reshape(-1)
    except (OSError, ValueError) as error:
        raise DistortionCorrectionError(
            f"Could not load calibration data from {path}: {error}"
        ) from error

    if camera_matrix.shape != (3, 3):
        raise DistortionCorrectionError(
            f"Invalid camera matrix shape {camera_matrix.shape}; expected (3, 3)."
        )
    if distortion_coefficients.size == 0:
        raise DistortionCorrectionError("Distortion coefficients are empty.")
    if saved_image_size.size != 2:
        raise DistortionCorrectionError(
            "Invalid saved image size; expected width and height."
        )
    if not np.all(np.isfinite(camera_matrix)) or not np.all(
        np.isfinite(distortion_coefficients)
    ):
        raise DistortionCorrectionError(
            "Calibration parameters contain non-finite values."
        )

    image_size = (int(saved_image_size[0]), int(saved_image_size[1]))
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise DistortionCorrectionError(
            f"Invalid saved calibration resolution: {image_size}."
        )

    return CalibrationData(
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        image_size=image_size,
    )


def undistort_image(
    image: NDArray[np.uint8],
    camera_matrix: NDArray[np.float64],
    distortion_coefficients: NDArray[np.float64],
) -> NDArray[np.uint8]:
    """Return a distortion-corrected copy of an image.

    This function is intentionally independent of file handling so later lane
    detection stages can reuse it for already-loaded images.
    """

    if image is None or image.size == 0:
        raise DistortionCorrectionError("Cannot undistort an empty image.")

    return cv2.undistort(
        image,
        camera_matrix,
        distortion_coefficients,
        None,
        camera_matrix,
    )


def add_label(image: NDArray[np.uint8], label: str) -> NDArray[np.uint8]:
    """Return a labelled copy for the side-by-side comparison."""

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


def write_image(path: Path, image: NDArray[np.uint8]) -> None:
    """Save an image or raise a clear error."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise DistortionCorrectionError(f"Could not write image: {path}")


def main() -> int:
    args = parse_args()
    input_path = project_path(args.input_image)
    calibration_path = project_path(args.calibration_file)
    output_dir = project_path(args.output_dir)

    try:
        calibration = load_calibration(calibration_path)

        if not input_path.is_file():
            raise DistortionCorrectionError(
                f"Input image not found: {input_path}"
            )
        original = cv2.imread(str(input_path))
        if original is None:
            raise DistortionCorrectionError(
                f"OpenCV could not decode the input image: {input_path}"
            )

        input_size = (original.shape[1], original.shape[0])
        if input_size != calibration.image_size:
            raise DistortionCorrectionError(
                f"Input image resolution {input_size[0]}x{input_size[1]} does not "
                f"match calibration resolution {calibration.image_size[0]}x"
                f"{calibration.image_size[1]}. Use an image captured at the "
                "calibrated resolution."
            )

        corrected = undistort_image(
            original,
            calibration.camera_matrix,
            calibration.distortion_coefficients,
        )

        undistorted_path = output_dir / f"{input_path.stem}_undistorted.jpg"
        comparison_path = output_dir / f"{input_path.stem}_comparison.jpg"
        comparison = cv2.hconcat(
            [add_label(original, "Original"), add_label(corrected, "Undistorted")]
        )

        write_image(undistorted_path, corrected)
        write_image(comparison_path, comparison)
    except DistortionCorrectionError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except cv2.error as error:
        print(f"OpenCV error: {error}", file=sys.stderr)
        return 1

    print(f"Input image: {input_path}")
    print(
        "Image resolution: "
        f"{calibration.image_size[0]}x{calibration.image_size[1]}"
    )
    print(f"Undistorted image saved to: {undistorted_path}")
    print(f"Side-by-side comparison saved to: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
