from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from .perspective import perspective_transform
    from .thresholding import combined_threshold
    from .undistort import load_calibration, scale_camera_matrix, undistort_image
except ImportError:
    from perspective import perspective_transform
    from thresholding import combined_threshold
    from undistort import load_calibration, scale_camera_matrix, undistort_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = PROJECT_ROOT / "calibration_data" / "calibration.npz"
DEFAULT_INPUT_IMAGE = PROJECT_ROOT / "test_images" / "straight_lines1.jpg"
OUTPUT_DIR = PROJECT_ROOT / "output_images" / "lane_detection"

NUMBER_OF_WINDOWS = 9
MARGIN_AT_1280_PIXELS = 70
MINIMUM_RECENTER_PIXELS = 50
MINIMUM_FIT_PIXELS = 50
ROBUST_SIGMA_MULTIPLIER = 3.0
MINIMUM_RESIDUAL_THRESHOLD_AT_1280 = 20.0
MINIMUM_VERTICAL_COVERAGE_RATIO = 0.40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect lane pixels and fit second-degree polynomials."
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        default=DEFAULT_INPUT_IMAGE,
        help="road image to process (default: test_images/straight_lines1.jpg)",
    )
    parser.add_argument(
        "--margin",
        type=int,
        choices=(70, 100),
        default=70,
        help="sliding-window margin at 1280-pixel width (default: 70)",
    )
    parser.add_argument(
        "--robust-refinement",
        action="store_true",
        help="reject large polynomial residuals and refit the lane curves",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    """Resolve relative command-line paths from the project root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def write_image(path: Path, image: np.ndarray) -> None:
    """Save an image or raise a clear error."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def histogram_starting_positions(binary_image: np.ndarray) -> tuple[int, int]:
    """Find the strongest left and right peaks in the lower-half histogram."""

    height, width = binary_image.shape
    histogram = np.count_nonzero(binary_image[height // 2 :, :], axis=0)
    midpoint = width // 2

    if histogram[:midpoint].max(initial=0) == 0:
        raise RuntimeError(
            "No left-lane pixels were found in the lower-half histogram."
        )
    if histogram[midpoint:].max(initial=0) == 0:
        raise RuntimeError(
            "No right-lane pixels were found in the lower-half histogram."
        )

    left_start = int(np.argmax(histogram[:midpoint]))
    right_start = int(np.argmax(histogram[midpoint:]) + midpoint)
    return left_start, right_start


def find_lane_pixels(
    binary_image: np.ndarray,
    number_of_windows: int = NUMBER_OF_WINDOWS,
    margin_at_1280: int = MARGIN_AT_1280_PIXELS,
    minimum_recenter_pixels: int = MINIMUM_RECENTER_PIXELS,
):
    """Find left- and right-lane pixels with a bottom-up sliding-window search."""

    if binary_image is None or binary_image.size == 0 or binary_image.ndim != 2:
        raise RuntimeError("Lane detection requires a non-empty binary image.")

    height, width = binary_image.shape
    left_current, right_current = histogram_starting_positions(binary_image)
    window_height = height // number_of_windows
    margin = max(1, int(round(margin_at_1280 * width / 1280)))

    nonzero_y, nonzero_x = binary_image.nonzero()
    left_indices = []
    right_indices = []
    windows = []

    for window_number in range(number_of_windows):
        y_high = height - window_number * window_height
        y_low = height - (window_number + 1) * window_height
        if window_number == number_of_windows - 1:
            y_low = 0

        left_x_low = max(0, left_current - margin)
        left_x_high = min(width, left_current + margin)
        right_x_low = max(0, right_current - margin)
        right_x_high = min(width, right_current + margin)

        left_window_indices = np.where(
            (nonzero_y >= y_low)
            & (nonzero_y < y_high)
            & (nonzero_x >= left_x_low)
            & (nonzero_x < left_x_high)
        )[0]
        right_window_indices = np.where(
            (nonzero_y >= y_low)
            & (nonzero_y < y_high)
            & (nonzero_x >= right_x_low)
            & (nonzero_x < right_x_high)
        )[0]

        left_indices.append(left_window_indices)
        right_indices.append(right_window_indices)
        windows.append(
            (
                (left_x_low, y_low, left_x_high, y_high),
                (right_x_low, y_low, right_x_high, y_high),
            )
        )

        if len(left_window_indices) >= minimum_recenter_pixels:
            left_current = int(np.mean(nonzero_x[left_window_indices]))
        if len(right_window_indices) >= minimum_recenter_pixels:
            right_current = int(np.mean(nonzero_x[right_window_indices]))

    left_indices = np.concatenate(left_indices)
    right_indices = np.concatenate(right_indices)

    left_x = nonzero_x[left_indices]
    left_y = nonzero_y[left_indices]
    right_x = nonzero_x[right_indices]
    right_y = nonzero_y[right_indices]
    return left_x, left_y, right_x, right_y, windows


def fit_lane_polynomials(
    left_x: np.ndarray,
    left_y: np.ndarray,
    right_x: np.ndarray,
    right_y: np.ndarray,
    image_height: int,
):
    """Fit x(y) polynomials and reject missing or crossing lane boundaries."""

    if len(left_x) < MINIMUM_FIT_PIXELS:
        raise RuntimeError(
            f"Cannot fit the left lane: found {len(left_x)} pixels, but at least "
            f"{MINIMUM_FIT_PIXELS} are required."
        )
    if len(right_x) < MINIMUM_FIT_PIXELS:
        raise RuntimeError(
            f"Cannot fit the right lane: found {len(right_x)} pixels, but at least "
            f"{MINIMUM_FIT_PIXELS} are required."
        )
    if len(np.unique(left_y)) < 3 or len(np.unique(right_y)) < 3:
        raise RuntimeError(
            "Cannot fit a second-degree polynomial without pixels at three or more "
            "different vertical positions."
        )

    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    plot_y = np.arange(image_height, dtype=np.float64)
    left_fit_x = np.polyval(left_fit, plot_y)
    right_fit_x = np.polyval(right_fit, plot_y)

    if np.any(left_fit_x >= right_fit_x):
        raise RuntimeError(
            "The fitted lane curves cross, so the detected pixels do not form a "
            "valid left/right lane pair."
        )

    return left_fit, right_fit, plot_y, left_fit_x, right_fit_x


def robust_inlier_mask(
    x: np.ndarray,
    y: np.ndarray,
    polynomial: np.ndarray,
    image_width: int,
) -> tuple[np.ndarray, float]:
    """Select pixels using a median-absolute-deviation residual threshold."""

    residuals = np.abs(x - np.polyval(polynomial, y))
    median_residual = float(np.median(residuals))
    median_absolute_deviation = float(
        np.median(np.abs(residuals - median_residual))
    )
    robust_sigma = 1.4826 * median_absolute_deviation
    minimum_threshold = MINIMUM_RESIDUAL_THRESHOLD_AT_1280 * image_width / 1280
    threshold = max(
        minimum_threshold,
        median_residual + ROBUST_SIGMA_MULTIPLIER * robust_sigma,
    )
    return residuals <= threshold, threshold


def validate_refined_pixels(
    lane_name: str,
    x: np.ndarray,
    y: np.ndarray,
    image_height: int,
) -> None:
    """Require enough refined pixels across a useful vertical image range."""

    if len(x) < MINIMUM_FIT_PIXELS:
        raise RuntimeError(
            f"Robust refinement for the {lane_name} lane retained {len(x)} "
            f"pixels, but at least {MINIMUM_FIT_PIXELS} are required."
        )
    if len(np.unique(y)) < 3:
        raise RuntimeError(
            f"Robust refinement for the {lane_name} lane did not retain pixels "
            "at three or more vertical positions."
        )

    vertical_coverage = (float(np.max(y)) - float(np.min(y)) + 1) / image_height
    if vertical_coverage < MINIMUM_VERTICAL_COVERAGE_RATIO:
        raise RuntimeError(
            f"Robust refinement for the {lane_name} lane retained only "
            f"{vertical_coverage:.1%} vertical coverage; at least "
            f"{MINIMUM_VERTICAL_COVERAGE_RATIO:.0%} is required."
        )


def robust_refine_polynomials(
    left_x: np.ndarray,
    left_y: np.ndarray,
    right_x: np.ndarray,
    right_y: np.ndarray,
    image_height: int,
    image_width: int,
) -> dict:
    """Fit, reject unusually large residuals, and refit both lane curves."""

    initial_left_fit, initial_right_fit, _, _, _ = fit_lane_polynomials(
        left_x,
        left_y,
        right_x,
        right_y,
        image_height,
    )
    left_inliers, left_threshold = robust_inlier_mask(
        left_x,
        left_y,
        initial_left_fit,
        image_width,
    )
    right_inliers, right_threshold = robust_inlier_mask(
        right_x,
        right_y,
        initial_right_fit,
        image_width,
    )

    refined_left_x = left_x[left_inliers]
    refined_left_y = left_y[left_inliers]
    refined_right_x = right_x[right_inliers]
    refined_right_y = right_y[right_inliers]
    validate_refined_pixels(
        "left", refined_left_x, refined_left_y, image_height
    )
    validate_refined_pixels(
        "right", refined_right_x, refined_right_y, image_height
    )

    left_fit, right_fit, plot_y, left_fit_x, right_fit_x = (
        fit_lane_polynomials(
            refined_left_x,
            refined_left_y,
            refined_right_x,
            refined_right_y,
            image_height,
        )
    )
    return {
        "left_x": refined_left_x,
        "left_y": refined_left_y,
        "right_x": refined_right_x,
        "right_y": refined_right_y,
        "left_fit": left_fit,
        "right_fit": right_fit,
        "plot_y": plot_y,
        "left_fit_x": left_fit_x,
        "right_fit_x": right_fit_x,
        "left_threshold": left_threshold,
        "right_threshold": right_threshold,
    }


def experiment_suffix(margin: int, robust_refinement: bool) -> str:
    """Return an empty suffix for baseline output and labels for experiments."""

    if margin == MARGIN_AT_1280_PIXELS and not robust_refinement:
        return ""

    parts = []
    if margin != MARGIN_AT_1280_PIXELS:
        parts.append(f"margin{margin}")
    if robust_refinement:
        parts.append("robust")
    return "_" + "_".join(parts)


def lane_width_diagnostics(
    left_fit: np.ndarray,
    right_fit: np.ndarray,
    image_height: int,
):
    """Estimate lane width in pixels at several vertical image positions."""

    bottom_y = image_height - 1
    sample_labels = ("20%", "50%", "80%", "bottom")
    sample_y = np.array(
        [
            round(0.20 * bottom_y),
            round(0.50 * bottom_y),
            round(0.80 * bottom_y),
            bottom_y,
        ],
        dtype=np.float64,
    )
    left_positions = np.polyval(left_fit, sample_y)
    right_positions = np.polyval(right_fit, sample_y)
    widths = right_positions - left_positions

    if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
        raise RuntimeError(
            "Lane-width diagnostics are unavailable because the fitted lane "
            "positions produce invalid width estimates."
        )

    minimum_width = float(np.min(widths))
    maximum_width = float(np.max(widths))
    mean_width = float(np.mean(widths))
    relative_variation = 100.0 * (maximum_width - minimum_width) / mean_width
    return (
        sample_labels,
        sample_y.astype(np.int32),
        widths,
        minimum_width,
        maximum_width,
        relative_variation,
    )


def binary_visualization(binary_image: np.ndarray) -> np.ndarray:
    """Convert a 0/1 binary image to a three-channel diagnostic image."""

    binary_255 = (binary_image * 255).astype(np.uint8)
    return cv2.cvtColor(binary_255, cv2.COLOR_GRAY2BGR)


def draw_sliding_windows(
    binary_image: np.ndarray,
    windows,
    left_x: np.ndarray,
    left_y: np.ndarray,
    right_x: np.ndarray,
    right_y: np.ndarray,
) -> np.ndarray:
    """Draw search windows and the detected lane pixels."""

    visualization = binary_visualization(binary_image)
    for left_window, right_window in windows:
        left_x_low, y_low, left_x_high, y_high = left_window
        right_x_low, _, right_x_high, _ = right_window
        cv2.rectangle(
            visualization,
            (left_x_low, y_low),
            (left_x_high, y_high),
            (0, 255, 0),
            2,
        )
        cv2.rectangle(
            visualization,
            (right_x_low, y_low),
            (right_x_high, y_high),
            (0, 255, 0),
            2,
        )

    visualization[left_y, left_x] = (255, 0, 0)
    visualization[right_y, right_x] = (0, 0, 255)
    return visualization


def curve_points(fit_x: np.ndarray, plot_y: np.ndarray, width: int) -> np.ndarray:
    """Return drawable polynomial points that fall inside the image."""

    valid = np.isfinite(fit_x) & (fit_x >= 0) & (fit_x < width)
    points = np.column_stack((fit_x[valid], plot_y[valid]))
    return np.round(points).astype(np.int32).reshape(-1, 1, 2)


def draw_polynomial_fit(
    binary_image: np.ndarray,
    left_x: np.ndarray,
    left_y: np.ndarray,
    right_x: np.ndarray,
    right_y: np.ndarray,
    plot_y: np.ndarray,
    left_fit_x: np.ndarray,
    right_fit_x: np.ndarray,
) -> np.ndarray:
    """Draw detected pixels and the fitted left and right curves."""

    visualization = binary_visualization(binary_image)
    visualization[left_y, left_x] = (255, 0, 0)
    visualization[right_y, right_x] = (0, 0, 255)

    width = binary_image.shape[1]
    left_curve = curve_points(left_fit_x, plot_y, width)
    right_curve = curve_points(right_fit_x, plot_y, width)
    if len(left_curve) > 1:
        cv2.polylines(visualization, [left_curve], False, (0, 255, 255), 6)
    if len(right_curve) > 1:
        cv2.polylines(visualization, [right_curve], False, (0, 255, 0), 6)
    return visualization


def main() -> int:
    args = parse_args()
    input_path = project_path(args.input_image)
    print(f"Input image: {input_path}")
    suffix = experiment_suffix(args.margin, args.robust_refinement)

    output_paths = {
        "warped_binary": OUTPUT_DIR
        / f"{input_path.stem}{suffix}_warped_binary.png",
        "sliding_windows": OUTPUT_DIR
        / f"{input_path.stem}{suffix}_sliding_windows.jpg",
        "polynomial_fit": OUTPUT_DIR
        / f"{input_path.stem}{suffix}_polynomial_fit.jpg",
    }

    try:
        calibration = load_calibration(CALIBRATION_FILE)
        if not input_path.is_file():
            raise RuntimeError(f"Input image not found: {input_path}")

        original = cv2.imread(str(input_path))
        if original is None:
            raise RuntimeError(f"OpenCV could not decode the image: {input_path}")

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
        binary_image = combined_threshold(undistorted)
        warped_binary, _, _, _, _ = perspective_transform(binary_image)

        write_image(
            output_paths["warped_binary"],
            (warped_binary * 255).astype(np.uint8),
        )

        left_x, left_y, right_x, right_y, windows = find_lane_pixels(
            warped_binary,
            margin_at_1280=args.margin,
        )
        print(f"Detected left-lane pixels: {len(left_x)}")
        print(f"Detected right-lane pixels: {len(right_x)}")
        sliding_windows_image = draw_sliding_windows(
            warped_binary,
            windows,
            left_x,
            left_y,
            right_x,
            right_y,
        )
        write_image(output_paths["sliding_windows"], sliding_windows_image)

        if args.robust_refinement:
            fit_result = robust_refine_polynomials(
                left_x,
                left_y,
                right_x,
                right_y,
                warped_binary.shape[0],
                warped_binary.shape[1],
            )
        else:
            left_fit, right_fit, plot_y, left_fit_x, right_fit_x = (
                fit_lane_polynomials(
                    left_x,
                    left_y,
                    right_x,
                    right_y,
                    warped_binary.shape[0],
                )
            )
            fit_result = {
                "left_x": left_x,
                "left_y": left_y,
                "right_x": right_x,
                "right_y": right_y,
                "left_fit": left_fit,
                "right_fit": right_fit,
                "plot_y": plot_y,
                "left_fit_x": left_fit_x,
                "right_fit_x": right_fit_x,
            }

        polynomial_image = draw_polynomial_fit(
            warped_binary,
            fit_result["left_x"],
            fit_result["left_y"],
            fit_result["right_x"],
            fit_result["right_y"],
            fit_result["plot_y"],
            fit_result["left_fit_x"],
            fit_result["right_fit_x"],
        )
        write_image(output_paths["polynomial_fit"], polynomial_image)
        width_diagnostics = lane_width_diagnostics(
            fit_result["left_fit"],
            fit_result["right_fit"],
            warped_binary.shape[0],
        )
    except (RuntimeError, OSError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Sliding-window margin at 1280-pixel width: {args.margin}")
    print(
        "Polynomial refinement: "
        + ("robust residual filtering" if args.robust_refinement else "baseline")
    )
    if args.robust_refinement:
        print(f"Retained left-lane pixels: {len(fit_result['left_x'])}")
        print(f"Retained right-lane pixels: {len(fit_result['right_x'])}")
        print(
            f"Left robust residual threshold: "
            f"{fit_result['left_threshold']:.2f} pixels"
        )
        print(
            f"Right robust residual threshold: "
            f"{fit_result['right_threshold']:.2f} pixels"
        )
    print(f"Left polynomial coefficients [A, B, C]: {fit_result['left_fit']}")
    print(f"Right polynomial coefficients [A, B, C]: {fit_result['right_fit']}")
    (
        sample_labels,
        sample_y,
        sampled_widths,
        minimum_width,
        maximum_width,
        relative_variation,
    ) = width_diagnostics
    print("Estimated lane width in image pixels (diagnostic only):")
    for label, y_position, width in zip(
        sample_labels,
        sample_y,
        sampled_widths,
    ):
        print(f"  {label} height (y={y_position}): {width:.2f} pixels")
    print(f"  Minimum sampled width: {minimum_width:.2f} pixels")
    print(f"  Maximum sampled width: {maximum_width:.2f} pixels")
    print(f"  Relative sampled-width variation: {relative_variation:.2f}%")
    print("  These pixel-width estimates are diagnostic, not rejection criteria.")
    for output_path in output_paths.values():
        print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
