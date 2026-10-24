"""Create the final lane-finding image with measurements.

Run this module from the project root:

    python src/final_output.py
    python src/final_output.py --input-image test_images/test1.jpg
    python src/final_output.py --all-images
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
    from .lane_measurements import (
        DEFAULT_LANE_WIDTH_METERS,
        DEFAULT_VISIBLE_ROAD_LENGTH_METERS,
        calculate_measurements,
        format_curvature,
        offset_direction,
    )
    from .lane_overlay import draw_lane_overlay
    from .perspective import perspective_transform
    from .thresholding import combined_threshold
    from .undistort import load_calibration, scale_camera_matrix, undistort_image
except ImportError:
    from lane_detection import find_lane_pixels, fit_lane_polynomials, write_image
    from lane_measurements import (
        DEFAULT_LANE_WIDTH_METERS,
        DEFAULT_VISIBLE_ROAD_LENGTH_METERS,
        calculate_measurements,
        format_curvature,
        offset_direction,
    )
    from lane_overlay import draw_lane_overlay
    from perspective import perspective_transform
    from thresholding import combined_threshold
    from undistort import load_calibration, scale_camera_matrix, undistort_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = PROJECT_ROOT / "calibration_data" / "calibration.npz"
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
DEFAULT_INPUT_IMAGE = PROJECT_ROOT / "test_images" / "straight_lines1.jpg"
OUTPUT_DIR = PROJECT_ROOT / "output_images" / "final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the lane area, curvature, and vehicle position."
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input-image",
        type=Path,
        default=DEFAULT_INPUT_IMAGE,
        help="road image to process (default: test_images/straight_lines1.jpg)",
    )
    input_group.add_argument(
        "--all-images",
        action="store_true",
        help="process every JPG image in test_images",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    """Resolve a relative command-line path from the project root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def draw_final_output(overlay: np.ndarray, measurements: dict) -> np.ndarray:
    """Add estimated curvature and vehicle-position text to a lane overlay."""

    if overlay is None or overlay.size == 0:
        raise RuntimeError("Cannot draw measurements on an empty image.")
    if overlay.ndim != 3 or overlay.shape[2] != 3:
        raise RuntimeError("Final output requires a three-channel road image.")

    offset = measurements["offset_meters"]
    direction = offset_direction(offset)
    if direction == "near the lane center":
        position_text = "Estimated vehicle position: near the lane center"
    else:
        position_text = (
            f"Estimated vehicle position: {abs(offset):.2f} m {direction}"
        )

    text_lines = (
        "Estimated left curvature: "
        f"{format_curvature(measurements['left_curvature'])}",
        "Estimated right curvature: "
        f"{format_curvature(measurements['right_curvature'])}",
        position_text,
    )

    font_scale = 0.75
    font_thickness = 2
    text_left = 40
    text_top = 58
    line_spacing = 38
    text_widths = [
        cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )[0][0]
        for text in text_lines
    ]

    result = overlay.copy()
    height, width = result.shape[:2]
    panel_left = 20
    panel_top = 20
    panel_right = min(width - 20, text_left + max(text_widths) + 20)
    panel_bottom = min(
        height - 20,
        text_top + (len(text_lines) - 1) * line_spacing + 20,
    )
    if panel_right <= panel_left or panel_bottom <= panel_top:
        raise RuntimeError("The image is too small for the measurement display.")
    if text_left + max(text_widths) > width - 20:
        raise RuntimeError("The image is too narrow for the measurement text.")

    darkened = result.copy()
    cv2.rectangle(
        darkened,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (0, 0, 0),
        -1,
    )
    panel_mask = np.zeros((height, width), dtype=bool)
    panel_mask[panel_top : panel_bottom + 1, panel_left : panel_right + 1] = True
    blended_panel = cv2.addWeighted(result, 0.35, darkened, 0.65, 0.0)
    result[panel_mask] = blended_panel[panel_mask]

    for index, text in enumerate(text_lines):
        cv2.putText(
            result,
            text,
            (text_left, text_top + index * line_spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return result


def process_image(input_path: Path, calibration) -> tuple[Path, dict]:
    """Process one road image and save its final annotated result."""

    output_path = OUTPUT_DIR / f"{input_path.stem}_final.jpg"

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
    binary = combined_threshold(undistorted)
    warped, _, inverse_matrix, _, _ = perspective_transform(binary)

    left_x, left_y, right_x, right_y, _ = find_lane_pixels(warped)
    left_fit, right_fit, plot_y, left_fit_x, right_fit_x = fit_lane_polynomials(
        left_x,
        left_y,
        right_x,
        right_y,
        warped.shape[0],
    )
    measurements = calculate_measurements(
        left_fit,
        right_fit,
        warped.shape[0],
        warped.shape[1],
        DEFAULT_LANE_WIDTH_METERS,
        DEFAULT_VISIBLE_ROAD_LENGTH_METERS,
    )
    overlay = draw_lane_overlay(
        undistorted,
        warped,
        inverse_matrix,
        plot_y,
        left_fit_x,
        right_fit_x,
    )
    final_image = draw_final_output(overlay, measurements)
    write_image(output_path, final_image)
    return output_path, measurements


def print_result(input_path: Path, output_path: Path, measurements: dict) -> None:
    """Print the measurements and saved path for one processed image."""

    print(f"Input image: {input_path}")
    print(
        "Estimated left curvature: "
        f"{format_curvature(measurements['left_curvature'])}"
    )
    print(
        "Estimated right curvature: "
        f"{format_curvature(measurements['right_curvature'])}"
    )
    print(
        f"Estimated vehicle position: {measurements['offset_meters']:+.3f} m "
        f"({offset_direction(measurements['offset_meters'])})"
    )
    print(f"Saved final output: {output_path}")


def main() -> int:
    args = parse_args()

    try:
        calibration = load_calibration(CALIBRATION_FILE)
    except (RuntimeError, OSError, ValueError, OverflowError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if not args.all_images:
        input_path = project_path(args.input_image)
        try:
            output_path, measurements = process_image(input_path, calibration)
        except (RuntimeError, OSError, ValueError, OverflowError, cv2.error) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        print_result(input_path, output_path, measurements)
        return 0

    image_paths = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    if not image_paths:
        print(f"Error: no JPG images found in {TEST_IMAGES_DIR}.", file=sys.stderr)
        return 1

    successful = 0
    failures = []
    for input_path in image_paths:
        try:
            output_path, measurements = process_image(input_path, calibration)
            print_result(input_path, output_path, measurements)
            successful += 1
        except (RuntimeError, OSError, ValueError, OverflowError, cv2.error) as error:
            failures.append((input_path.name, str(error)))
            print(f"Failed: {input_path.name}: {error}", file=sys.stderr)

    print("\nFinal-output summary")
    print(f"Total images found: {len(image_paths)}")
    print(f"Successfully processed images: {successful}")
    print(f"Failed images: {len(failures)}")
    for filename, reason in failures:
        print(f"- {filename}: {reason}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
