"""Calibrate the project camera and create distortion-correction examples.

Run this module from the project root:

    python src/calibration.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "camera_cal"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output_images" / "calibration"
DEFAULT_CALIBRATION_FILE = PROJECT_ROOT / "calibration_data" / "calibration.npz"

FloatArray = NDArray[np.float32]


class CalibrationError(RuntimeError):
    """Raised when calibration cannot be completed with the supplied inputs."""


@dataclass(frozen=True)
class DetectionResult:
    """Corresponding points and metadata collected from calibration images."""

    object_points: list[FloatArray]
    image_points: list[FloatArray]
    image_size: tuple[int, int]
    processed_count: int
    successful_images: list[Path]


@dataclass(frozen=True)
class CalibrationResult:
    """Values returned by OpenCV camera calibration."""

    rms_error: float
    camera_matrix: NDArray[np.float64]
    distortion_coefficients: NDArray[np.float64]
    rotation_vectors: Sequence[NDArray[np.float64]]
    translation_vectors: Sequence[NDArray[np.float64]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the camera from chessboard photographs."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="directory containing calibration images (default: camera_cal)",
    )
    parser.add_argument(
        "--image-glob",
        default="calibration*.jpg",
        help="file pattern within --images-dir (default: calibration*.jpg)",
    )
    parser.add_argument(
        "--corners-x",
        type=int,
        default=9,
        help="number of internal chessboard corners horizontally (default: 9)",
    )
    parser.add_argument(
        "--corners-y",
        type=int,
        default=6,
        help="number of internal chessboard corners vertically (default: 6)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for generated diagnostic images",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=DEFAULT_CALIBRATION_FILE,
        help="output .npz file for camera parameters",
    )
    parser.add_argument(
        "--example-image",
        type=Path,
        default=None,
        help="image used for undistortion examples (default: first successful image)",
    )
    args = parser.parse_args()

    if args.corners_x <= 1 or args.corners_y <= 1:
        parser.error("--corners-x and --corners-y must both be greater than 1")

    return args


def project_path(path: Path) -> Path:
    """Resolve relative command-line paths from the project root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def natural_sort_key(path: Path) -> tuple[object, ...]:
    """Sort names containing numbers in human order (2 before 10)."""

    parts: list[object] = []
    current = ""
    is_digit: bool | None = None

    for character in path.name.lower():
        character_is_digit = character.isdigit()
        if is_digit is None or character_is_digit == is_digit:
            current += character
        else:
            parts.append(int(current) if is_digit else current)
            current = character
        is_digit = character_is_digit

    if current:
        parts.append(int(current) if is_digit else current)
    return tuple(parts)


def make_object_points(pattern_size: tuple[int, int]) -> FloatArray:
    """Create planar (x, y, 0) coordinates for one chessboard view."""

    columns, rows = pattern_size
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    return points


def write_image(path: Path, image: NDArray[np.uint8]) -> None:
    """Write an image and report a useful error if OpenCV cannot save it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise CalibrationError(f"Could not write image: {path}")


def collect_calibration_points(
    image_paths: Sequence[Path],
    pattern_size: tuple[int, int],
    corners_output_dir: Path,
) -> DetectionResult:
    """Detect and refine chessboard corners in all readable input images."""

    object_template = make_object_points(pattern_size)
    object_points: list[FloatArray] = []
    image_points: list[FloatArray] = []
    successful_images: list[Path] = []
    image_size: tuple[int, int] | None = None
    processed_count = 0

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    detection_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: could not read {image_path}; skipping it.", file=sys.stderr)
            continue

        processed_count += 1
        current_size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            print(
                f"Warning: {image_path} has size {current_size}, expected "
                f"{image_size}; skipping it.",
                file=sys.stderr,
            )
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size, flags=detection_flags
        )
        if not found:
            print(f"Warning: chessboard corners not found in {image_path.name}.")
            continue

        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria,
        )
        object_points.append(object_template.copy())
        image_points.append(refined_corners)
        successful_images.append(image_path)

        annotated = image.copy()
        cv2.drawChessboardCorners(annotated, pattern_size, refined_corners, found)
        write_image(
            corners_output_dir / f"{image_path.stem}_corners.jpg",
            annotated,
        )
        if len(successful_images) == 1:
            write_image(corners_output_dir / "corners_example.jpg", annotated)

    if image_size is None:
        raise CalibrationError("None of the calibration images could be read.")

    return DetectionResult(
        object_points=object_points,
        image_points=image_points,
        image_size=image_size,
        processed_count=processed_count,
        successful_images=successful_images,
    )


def calibrate(detections: DetectionResult) -> CalibrationResult:
    """Calculate intrinsic camera parameters from corresponding points."""

    if not detections.object_points:
        raise CalibrationError(
            "No complete chessboard pattern was detected. Check the corner dimensions "
            "and try --corners-x/--corners-y with the correct internal-corner counts."
        )

    rms_error, camera_matrix, distortion_coefficients, rvecs, tvecs = (
        cv2.calibrateCamera(
            detections.object_points,
            detections.image_points,
            detections.image_size,
            None,
            None,
        )
    )

    return CalibrationResult(
        rms_error=float(rms_error),
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        rotation_vectors=rvecs,
        translation_vectors=tvecs,
    )


def save_calibration(
    path: Path,
    calibration: CalibrationResult,
    detections: DetectionResult,
    pattern_size: tuple[int, int],
) -> None:
    """Persist camera parameters and calibration metadata in NumPy format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        camera_matrix=calibration.camera_matrix,
        distortion_coefficients=calibration.distortion_coefficients,
        rotation_vectors=np.asarray(calibration.rotation_vectors),
        translation_vectors=np.asarray(calibration.translation_vectors),
        rms_reprojection_error=np.asarray(calibration.rms_error),
        image_size=np.asarray(detections.image_size),
        pattern_size=np.asarray(pattern_size),
        successful_images=np.asarray(
            [image.name for image in detections.successful_images]
        ),
    )


def create_undistortion_examples(
    image_path: Path,
    output_dir: Path,
    calibration: CalibrationResult,
) -> tuple[Path, Path]:
    """Save an undistorted image and a labelled original/undistorted comparison."""

    original = cv2.imread(str(image_path))
    if original is None:
        raise CalibrationError(f"Could not read example image: {image_path}")

    height, width = original.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        calibration.camera_matrix,
        calibration.distortion_coefficients,
        (width, height),
        alpha=1,
        newImgSize=(width, height),
    )
    undistorted = cv2.undistort(
        original,
        calibration.camera_matrix,
        calibration.distortion_coefficients,
        None,
        new_camera_matrix,
    )

    undistorted_path = output_dir / "undistorted_example.jpg"
    write_image(undistorted_path, undistorted)

    labelled_original = original.copy()
    labelled_undistorted = undistorted.copy()
    label_style = {
        "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
        "fontScale": 1.0,
        "color": (0, 255, 0),
        "thickness": 2,
        "lineType": cv2.LINE_AA,
    }
    cv2.putText(labelled_original, "Original", (30, 45), **label_style)
    cv2.putText(labelled_undistorted, "Undistorted", (30, 45), **label_style)
    comparison = cv2.hconcat([labelled_original, labelled_undistorted])

    comparison_path = output_dir / "original_vs_undistorted.jpg"
    write_image(comparison_path, comparison)
    return undistorted_path, comparison_path


def main() -> int:
    args = parse_args()
    images_dir = project_path(args.images_dir)
    output_dir = project_path(args.output_dir)
    calibration_file = project_path(args.calibration_file)
    pattern_size = (args.corners_x, args.corners_y)

    image_paths = sorted(images_dir.glob(args.image_glob), key=natural_sort_key)
    if not image_paths:
        print(
            f"Error: no images matching {args.image_glob!r} found in {images_dir}.",
            file=sys.stderr,
        )
        return 1

    try:
        detections = collect_calibration_points(
            image_paths,
            pattern_size,
            output_dir / "corners",
        )
        print(f"Calibration images processed: {detections.processed_count}")
        print(f"Successful corner detections: {len(detections.successful_images)}")

        calibration = calibrate(detections)
        save_calibration(
            calibration_file,
            calibration,
            detections,
            pattern_size,
        )

        example_image = (
            project_path(args.example_image)
            if args.example_image is not None
            else detections.successful_images[0]
        )
        undistorted_path, comparison_path = create_undistortion_examples(
            example_image,
            output_dir,
            calibration,
        )
    except CalibrationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except cv2.error as error:
        print(f"OpenCV error: {error}", file=sys.stderr)
        return 1

    print("Camera matrix:")
    print(calibration.camera_matrix)
    print("Distortion coefficients:")
    print(calibration.distortion_coefficients)
    print(f"RMS reprojection error: {calibration.rms_error:.6f}")
    print(f"Calibration parameters saved to: {calibration_file}")
    print(f"Undistorted example saved to: {undistorted_path}")
    print(f"Side-by-side comparison saved to: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
