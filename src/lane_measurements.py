"""Estimate lane curvature and vehicle offset from a road image.

The metric results use assumed lane dimensions rather than a physically
verified camera-to-world calibration, so they should be treated as estimates.

Run this module from the project root:

    python src/lane_measurements.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from .lane_detection import (
        draw_polynomial_fit,
        find_lane_pixels,
        fit_lane_polynomials,
        lane_width_diagnostics,
        write_image,
    )
    from .perspective import perspective_transform
    from .thresholding import combined_threshold
    from .undistort import load_calibration, scale_camera_matrix, undistort_image
except ImportError:
    from lane_detection import (
        draw_polynomial_fit,
        find_lane_pixels,
        fit_lane_polynomials,
        lane_width_diagnostics,
        write_image,
    )
    from perspective import perspective_transform
    from thresholding import combined_threshold
    from undistort import load_calibration, scale_camera_matrix, undistort_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = PROJECT_ROOT / "calibration_data" / "calibration.npz"
DEFAULT_INPUT_IMAGE = PROJECT_ROOT / "test_images" / "straight_lines1.jpg"
OUTPUT_DIR = PROJECT_ROOT / "output_images" / "lane_measurements"

DEFAULT_LANE_WIDTH_METERS = 3.7
DEFAULT_VISIBLE_ROAD_LENGTH_METERS = 30.0
WIDTH_VARIATION_WARNING_PERCENT = 10.0
STRAIGHT_LINE_COEFFICIENT_TOLERANCE = 1e-12
CENTER_OFFSET_TOLERANCE_METERS = 0.01


def positive_float(value: str) -> float:
    """Parse a finite, positive command-line number."""

    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate lane curvature and vehicle offset in meters."
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        default=DEFAULT_INPUT_IMAGE,
        help="road image to process (default: test_images/straight_lines1.jpg)",
    )
    parser.add_argument(
        "--lane-width-meters",
        type=positive_float,
        default=DEFAULT_LANE_WIDTH_METERS,
        help="assumed physical lane width (default: 3.7)",
    )
    parser.add_argument(
        "--visible-road-length-meters",
        type=positive_float,
        default=DEFAULT_VISIBLE_ROAD_LENGTH_METERS,
        help="assumed road length represented by the warped image (default: 30)",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    """Resolve a relative command-line path from the project root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def validated_polynomial(polynomial: np.ndarray, lane_name: str) -> np.ndarray:
    """Return three finite polynomial coefficients or report invalid input."""

    coefficients = np.asarray(polynomial, dtype=np.float64).reshape(-1)
    if coefficients.size != 3 or not np.all(np.isfinite(coefficients)):
        raise RuntimeError(
            f"The {lane_name} lane polynomial must contain three finite coefficients."
        )
    return coefficients


def curvature_radius_meters(
    polynomial: np.ndarray,
    xm_per_pix: float,
    ym_per_pix: float,
    y_eval_pixels: float,
) -> float:
    """Calculate curvature after converting an x(y) polynomial to meters."""

    quadratic, linear, constant = polynomial
    metric_quadratic = quadratic * xm_per_pix / (ym_per_pix**2)
    metric_linear = linear * xm_per_pix / ym_per_pix
    metric_constant = constant * xm_per_pix
    metric_coefficients = np.array(
        [metric_quadratic, metric_linear, metric_constant],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(metric_coefficients)):
        raise RuntimeError("Polynomial conversion to meters produced invalid values.")

    if abs(metric_quadratic) <= STRAIGHT_LINE_COEFFICIENT_TOLERANCE:
        return float("inf")

    y_eval_meters = y_eval_pixels * ym_per_pix
    slope = 2.0 * metric_quadratic * y_eval_meters + metric_linear
    radius = (1.0 + slope**2) ** 1.5 / abs(2.0 * metric_quadratic)
    if not np.isfinite(radius) or radius <= 0:
        raise RuntimeError("Curvature calculation produced an invalid radius.")
    return float(radius)


def calculate_measurements(
    left_fit: np.ndarray,
    right_fit: np.ndarray,
    image_height: int,
    image_width: int,
    lane_width_meters: float,
    visible_road_length_meters: float,
) -> dict:
    """Calculate approximate lane curvature, scale, and vehicle offset values."""

    if image_height <= 0 or image_width <= 0:
        raise RuntimeError("Image dimensions must be positive.")
    if not np.isfinite(lane_width_meters) or lane_width_meters <= 0:
        raise RuntimeError("Assumed lane width must be finite and positive.")
    if (
        not np.isfinite(visible_road_length_meters)
        or visible_road_length_meters <= 0
    ):
        raise RuntimeError("Assumed visible road length must be finite and positive.")

    left_fit = validated_polynomial(left_fit, "left")
    right_fit = validated_polynomial(right_fit, "right")
    bottom_y = float(image_height - 1)
    left_bottom = float(np.polyval(left_fit, bottom_y))
    right_bottom = float(np.polyval(right_fit, bottom_y))
    if not np.isfinite(left_bottom) or not np.isfinite(right_bottom):
        raise RuntimeError("Lane positions at the bottom of the image are invalid.")

    lane_width_pixels = right_bottom - left_bottom
    if not np.isfinite(lane_width_pixels) or lane_width_pixels <= 0:
        raise RuntimeError(
            "Estimated lane width at the bottom must be finite and positive."
        )

    xm_per_pix = lane_width_meters / lane_width_pixels
    ym_per_pix = visible_road_length_meters / image_height
    if (
        not np.isfinite(xm_per_pix)
        or not np.isfinite(ym_per_pix)
        or xm_per_pix <= 0
        or ym_per_pix <= 0
    ):
        raise RuntimeError("Pixel-to-meter conversion produced an invalid scale.")

    left_curvature = curvature_radius_meters(
        left_fit,
        xm_per_pix,
        ym_per_pix,
        bottom_y,
    )
    right_curvature = curvature_radius_meters(
        right_fit,
        xm_per_pix,
        ym_per_pix,
        bottom_y,
    )

    vehicle_center_pixels = image_width / 2.0
    lane_center_pixels = (left_bottom + right_bottom) / 2.0
    offset_meters = (vehicle_center_pixels - lane_center_pixels) * xm_per_pix
    if not np.isfinite(offset_meters):
        raise RuntimeError("Vehicle-offset calculation produced an invalid value.")

    return {
        "left_bottom": left_bottom,
        "right_bottom": right_bottom,
        "lane_width_pixels": lane_width_pixels,
        "lane_center_pixels": lane_center_pixels,
        "vehicle_center_pixels": vehicle_center_pixels,
        "xm_per_pix": xm_per_pix,
        "ym_per_pix": ym_per_pix,
        "left_curvature": left_curvature,
        "right_curvature": right_curvature,
        "offset_meters": float(offset_meters),
    }


def offset_direction(offset_meters: float) -> str:
    """Describe the vehicle position using the documented sign convention."""

    if abs(offset_meters) <= CENTER_OFFSET_TOLERANCE_METERS:
        return "near the lane center"
    if offset_meters > 0:
        return "right of the estimated lane center"
    return "left of the estimated lane center"


def format_curvature(radius: float) -> str:
    if np.isinf(radius):
        return "infinite (approximately straight)"
    return f"{radius:.2f} m"


def draw_measurements(
    warped_binary: np.ndarray,
    left_x: np.ndarray,
    left_y: np.ndarray,
    right_x: np.ndarray,
    right_y: np.ndarray,
    plot_y: np.ndarray,
    left_fit_x: np.ndarray,
    right_fit_x: np.ndarray,
    measurements: dict,
) -> np.ndarray:
    """Draw fitted curves, bottom centers, curvature, and vehicle offset."""

    visualization = draw_polynomial_fit(
        warped_binary,
        left_x,
        left_y,
        right_x,
        right_y,
        plot_y,
        left_fit_x,
        right_fit_x,
    )
    height, width = warped_binary.shape
    lane_center_x = int(round(measurements["lane_center_pixels"]))
    vehicle_center_x = int(round(measurements["vehicle_center_pixels"]))
    lane_center_x = int(np.clip(lane_center_x, 0, width - 1))
    vehicle_center_x = int(np.clip(vehicle_center_x, 0, width - 1))
    marker_top = max(0, height - 90)

    cv2.line(
        visualization,
        (lane_center_x, height - 1),
        (lane_center_x, marker_top),
        (255, 0, 255),
        5,
    )
    cv2.line(
        visualization,
        (vehicle_center_x, height - 1),
        (vehicle_center_x, marker_top),
        (255, 255, 0),
        5,
    )

    direction = offset_direction(measurements["offset_meters"])
    text_lines = (
        f"Estimated left curvature: {format_curvature(measurements['left_curvature'])}",
        f"Estimated right curvature: {format_curvature(measurements['right_curvature'])}",
        f"Estimated vehicle offset: {measurements['offset_meters']:+.3f} m",
        f"Vehicle position: {direction}",
        "Magenta: lane center | Cyan: vehicle center",
    )
    font_scale = 0.65 if width >= 1000 else 0.52
    line_spacing = 30 if height >= 700 else 25
    cv2.rectangle(
        visualization,
        (10, 8),
        (min(width - 10, 750), 18 + line_spacing * len(text_lines)),
        (0, 0, 0),
        -1,
    )
    for line_number, line in enumerate(text_lines, start=1):
        cv2.putText(
            visualization,
            line,
            (20, 8 + line_number * line_spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return visualization


def main() -> int:
    args = parse_args()
    input_path = project_path(args.input_image)
    output_path = OUTPUT_DIR / f"{input_path.stem}_measurements.jpg"

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
        warped_binary, _, _, _, _ = perspective_transform(binary_image)

        left_x, left_y, right_x, right_y, _ = find_lane_pixels(warped_binary)
        left_fit, right_fit, plot_y, left_fit_x, right_fit_x = (
            fit_lane_polynomials(
                left_x,
                left_y,
                right_x,
                right_y,
                warped_binary.shape[0],
            )
        )
        width_diagnostics = lane_width_diagnostics(
            left_fit,
            right_fit,
            warped_binary.shape[0],
        )
        measurements = calculate_measurements(
            left_fit,
            right_fit,
            warped_binary.shape[0],
            warped_binary.shape[1],
            args.lane_width_meters,
            args.visible_road_length_meters,
        )
        diagnostic_image = draw_measurements(
            warped_binary,
            left_x,
            left_y,
            right_x,
            right_y,
            plot_y,
            left_fit_x,
            right_fit_x,
            measurements,
        )
        write_image(output_path, diagnostic_image)
    except (RuntimeError, OSError, ValueError, OverflowError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    (
        _,
        _,
        _,
        minimum_sampled_width,
        maximum_sampled_width,
        relative_width_variation,
    ) = width_diagnostics
    direction = offset_direction(measurements["offset_meters"])

    print("Estimated lane measurements")
    print(f"Input image: {input_path.name}")
    print(f"Image resolution: {image_width}x{image_height}")
    print(
        "Left curvature radius: "
        f"{format_curvature(measurements['left_curvature'])}"
    )
    print(
        "Right curvature radius: "
        f"{format_curvature(measurements['right_curvature'])}"
    )
    print(f"Vehicle offset: {measurements['offset_meters']:+.3f} m (estimated)")
    print(f"Offset direction: {direction}")
    print(
        "Estimated lane width at bottom: "
        f"{measurements['lane_width_pixels']:.2f} pixels"
    )
    print(f"Horizontal scale: {measurements['xm_per_pix']:.6f} m/pixel")
    print(f"Vertical scale: {measurements['ym_per_pix']:.6f} m/pixel")
    print(f"Assumed physical lane width: {args.lane_width_meters:.2f} m")
    print(
        "Assumed visible road length: "
        f"{args.visible_road_length_meters:.2f} m"
    )
    print(
        "Sampled lane-width range: "
        f"{minimum_sampled_width:.2f}-{maximum_sampled_width:.2f} pixels"
    )
    print(f"Sampled lane-width variation: {relative_width_variation:.2f}%")
    if relative_width_variation > WIDTH_VARIATION_WARNING_PERCENT:
        print(
            "Warning: sampled lane-width variation exceeds 10%; the estimated "
            "metric measurements may be unreliable.",
            file=sys.stderr,
        )
    print(
        "Scale limitation: values assume a fixed lane width and visible road "
        "length; they are not based on a verified camera-to-world calibration."
    )
    print(f"Saved diagnostic image: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
