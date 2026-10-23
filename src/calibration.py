

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from .undistort import undistort_image
except ImportError:
    from undistort import undistort_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "camera_cal"
IMAGE_PATTERN = "calibration*.jpg"
OUTPUT_DIR = PROJECT_ROOT / "output_images" / "calibration"
CALIBRATION_FILE = PROJECT_ROOT / "calibration_data" / "calibration.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the camera from chessboard photographs."
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
    args = parser.parse_args()

    if args.corners_x <= 1 or args.corners_y <= 1:
        parser.error("--corners-x and --corners-y must both be greater than 1")

    return args


def calibration_image_number(path: Path) -> int:
    """Return the number from a name such as calibration12.jpg."""

    return int(path.stem.removeprefix("calibration"))


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def collect_calibration_points(
    image_paths,
    pattern_size: tuple[int, int],
    corners_output_dir: Path,
):
    """Detect and refine chessboard corners in all readable input images."""

    columns, rows = pattern_size
    object_template = np.zeros((columns * rows, 3), dtype=np.float32)
    object_template[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

    object_points = []
    image_points = []
    successful_images = []
    image_size = None
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
        raise RuntimeError("None of the calibration images could be read.")

    return (
        object_points,
        image_points,
        image_size,
        processed_count,
        successful_images,
    )


def calibrate(object_points, image_points, image_size):
    """Calculate intrinsic camera parameters from corresponding points."""

    if not object_points:
        raise RuntimeError(
            "No complete chessboard pattern was detected. Check the corner dimensions "
            "and try --corners-x/--corners-y with the correct internal-corner counts."
        )

    return cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )


def save_calibration(
    path: Path,
    rms_error,
    camera_matrix,
    distortion_coefficients,
    rotation_vectors,
    translation_vectors,
    image_size,
    pattern_size: tuple[int, int],
    successful_images,
) -> None:
    """Persist camera parameters and calibration metadata in NumPy format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        rotation_vectors=np.asarray(rotation_vectors),
        translation_vectors=np.asarray(translation_vectors),
        rms_reprojection_error=np.asarray(rms_error),
        image_size=np.asarray(image_size),
        pattern_size=np.asarray(pattern_size),
        successful_images=np.asarray([image.name for image in successful_images]),
    )


def create_undistortion_examples(
    image_path: Path,
    output_dir: Path,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> tuple[Path, Path]:
    """Save an undistorted image and a labelled original/undistorted comparison."""

    original = cv2.imread(str(image_path))
    if original is None:
        raise RuntimeError(f"Could not read example image: {image_path}")

    undistorted = undistort_image(
        original,
        camera_matrix,
        distortion_coefficients,
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
    pattern_size = (args.corners_x, args.corners_y)

    image_paths = sorted(IMAGES_DIR.glob(IMAGE_PATTERN), key=calibration_image_number)
    if not image_paths:
        print(
            f"Error: no images matching {IMAGE_PATTERN!r} found in {IMAGES_DIR}.",
            file=sys.stderr,
        )
        return 1

    try:
        (
            object_points,
            image_points,
            image_size,
            processed_count,
            successful_images,
        ) = collect_calibration_points(
            image_paths,
            pattern_size,
            OUTPUT_DIR / "corners",
        )
        print(f"Calibration images processed: {processed_count}")
        print(f"Successful corner detections: {len(successful_images)}")

        (
            rms_error,
            camera_matrix,
            distortion_coefficients,
            rotation_vectors,
            translation_vectors,
        ) = calibrate(object_points, image_points, image_size)
        save_calibration(
            CALIBRATION_FILE,
            rms_error,
            camera_matrix,
            distortion_coefficients,
            rotation_vectors,
            translation_vectors,
            image_size,
            pattern_size,
            successful_images,
        )

        undistorted_path, comparison_path = create_undistortion_examples(
            successful_images[0],
            OUTPUT_DIR,
            camera_matrix,
            distortion_coefficients,
        )
    except (RuntimeError, OSError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print("Camera matrix:")
    print(camera_matrix)
    print("Distortion coefficients:")
    print(distortion_coefficients)
    print(f"RMS reprojection error: {rms_error:.6f}")
    print(f"Calibration parameters saved to: {CALIBRATION_FILE}")
    print(f"Undistorted example saved to: {undistorted_path}")
    print(f"Side-by-side comparison saved to: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
