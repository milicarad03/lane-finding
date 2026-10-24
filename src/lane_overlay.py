"""Draw the detected lane area on an undistorted road image.

Run this module from the project root:

    python src/lane_overlay.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from .lane_detection import (
        find_lane_pixels,
        fit_lane_polynomials,
        write_image,
    )
    from .perspective import perspective_transform
    from .thresholding import combined_threshold
    from .undistort import load_calibration, scale_camera_matrix, undistort_image
except ImportError:
    from lane_detection import find_lane_pixels, fit_lane_polynomials, write_image
    from perspective import perspective_transform
    from thresholding import combined_threshold
    from undistort import load_calibration, scale_camera_matrix, undistort_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = PROJECT_ROOT / "calibration_data" / "calibration.npz"
DEFAULT_INPUT_IMAGE = PROJECT_ROOT / "test_images" / "straight_lines1.jpg"
OUTPUT_DIR = PROJECT_ROOT / "output_images" / "lane_overlay"
OVERLAY_WEIGHT = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the detected lane area on an undistorted road image."
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        default=DEFAULT_INPUT_IMAGE,
        help="road image to process (default: test_images/straight_lines1.jpg)",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    """Resolve a relative command-line path from the project root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def create_lane_polygon(
    image_shape: tuple[int, int],
    plot_y: np.ndarray,
    left_fit_x: np.ndarray,
    right_fit_x: np.ndarray,
) -> np.ndarray:
    """Return a green bird's-eye image filled between valid lane boundaries."""

    height, width = image_shape
    plot_y = np.asarray(plot_y, dtype=np.float64).reshape(-1)
    left_fit_x = np.asarray(left_fit_x, dtype=np.float64).reshape(-1)
    right_fit_x = np.asarray(right_fit_x, dtype=np.float64).reshape(-1)

    if height <= 0 or width <= 0:
        raise RuntimeError("Lane polygon requires positive image dimensions.")
    if (
        plot_y.size != height
        or left_fit_x.size != height
        or right_fit_x.size != height
    ):
        raise RuntimeError(
            "Fitted lane boundaries must contain one point for each image row."
        )
    if not (
        np.all(np.isfinite(plot_y))
        and np.all(np.isfinite(left_fit_x))
        and np.all(np.isfinite(right_fit_x))
    ):
        raise RuntimeError("Fitted lane boundaries contain invalid coordinates.")

    lane_width = right_fit_x - left_fit_x
    if np.any(lane_width <= 0):
        raise RuntimeError(
            "Invalid lane geometry: the fitted boundaries cross or overlap."
        )

    left_points = np.column_stack((left_fit_x, plot_y))
    right_points = np.column_stack((right_fit_x, plot_y))[::-1]
    polygon = np.round(np.vstack((left_points, right_points))).astype(np.int32)

    lane_area = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.fillPoly(lane_area, [polygon.reshape(-1, 1, 2)], (0, 255, 0))
    if not np.any(lane_area):
        raise RuntimeError("The fitted lane polygon does not intersect the image.")
    return lane_area


def draw_lane_overlay(
    undistorted: np.ndarray,
    warped_binary: np.ndarray,
    inverse_matrix: np.ndarray,
    plot_y: np.ndarray,
    left_fit_x: np.ndarray,
    right_fit_x: np.ndarray,
) -> np.ndarray:
    """Project the bird's-eye lane polygon back and blend it with the road."""

    if undistorted is None or undistorted.size == 0:
        raise RuntimeError("Cannot draw a lane overlay on an empty road image.")
    if undistorted.ndim != 3 or undistorted.shape[2] != 3:
        raise RuntimeError("Lane overlay requires a three-channel road image.")
    if warped_binary.ndim != 2:
        raise RuntimeError("Lane overlay requires a two-dimensional warped mask.")
    if warped_binary.shape != undistorted.shape[:2]:
        raise RuntimeError(
            "The warped binary and undistorted road image must have equal sizes."
        )

    height, width = warped_binary.shape
    lane_area = create_lane_polygon(
        (height, width),
        plot_y,
        left_fit_x,
        right_fit_x,
    )
    return blend_lane_overlay(undistorted, lane_area, inverse_matrix)


def blend_lane_overlay(
    undistorted: np.ndarray,
    lane_area: np.ndarray,
    inverse_matrix: np.ndarray,
) -> np.ndarray:
    """Project and blend a lane area only where its projected mask is present."""

    height, width = undistorted.shape[:2]
    projected_lane = cv2.warpPerspective(
        lane_area,
        inverse_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
    )
    lane_mask = np.any(lane_area != 0, axis=2).astype(np.uint8)
    projected_mask = cv2.warpPerspective(
        lane_mask,
        inverse_matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)

    blended = cv2.addWeighted(
        undistorted,
        1.0 - OVERLAY_WEIGHT,
        projected_lane,
        OVERLAY_WEIGHT,
        0.0,
    )
    result = undistorted.copy()
    result[projected_mask] = blended[projected_mask]
    return result


def main() -> int:
    args = parse_args()
    input_path = project_path(args.input_image)
    output_path = OUTPUT_DIR / f"{input_path.stem}_lane_overlay.jpg"

    try:
        calibration = load_calibration(CALIBRATION_FILE)
        if not input_path.is_file():
            raise RuntimeError(f"Input image not found: {input_path}")

        original = cv2.imread(str(input_path))
        if original is None:
            raise RuntimeError(f"OpenCV could not decode the image: {input_path}")

        image_height, image_width = original.shape[:2]
        input_size = (image_width, image_height)
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
        binary_image = combined_threshold(undistorted)
        warped_binary, _, inverse_matrix, _, _ = perspective_transform(binary_image)

        left_x, left_y, right_x, right_y, _ = find_lane_pixels(warped_binary)
        _, _, plot_y, left_fit_x, right_fit_x = fit_lane_polynomials(
            left_x,
            left_y,
            right_x,
            right_y,
            warped_binary.shape[0],
        )
        result = draw_lane_overlay(
            undistorted,
            warped_binary,
            inverse_matrix,
            plot_y,
            left_fit_x,
            right_fit_x,
        )
        write_image(output_path, result)
    except (RuntimeError, OSError, ValueError, OverflowError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Input image: {input_path}")
    print(f"Saved lane overlay: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
