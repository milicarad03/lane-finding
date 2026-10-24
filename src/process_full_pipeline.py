"""Process every JPG test image through Steps 2-7 of the pipeline.

Run this module from the project root:

    python src/process_full_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from .lane_detection import (
        draw_polynomial_fit,
        draw_sliding_windows,
        find_lane_pixels,
        fit_lane_polynomials,
        lane_width_diagnostics,
        write_image,
    )
    from .lane_measurements import (
        DEFAULT_LANE_WIDTH_METERS,
        DEFAULT_VISIBLE_ROAD_LENGTH_METERS,
        WIDTH_VARIATION_WARNING_PERCENT,
        calculate_measurements,
        draw_measurements,
        format_curvature,
        offset_direction,
    )
    from .lane_overlay import draw_lane_overlay
    from .perspective import perspective_transform
    from .thresholding import combined_threshold, mask_for_saving
    from .undistort import load_calibration, scale_camera_matrix, undistort_image
except ImportError:
    from lane_detection import (
        draw_polynomial_fit,
        draw_sliding_windows,
        find_lane_pixels,
        fit_lane_polynomials,
        lane_width_diagnostics,
        write_image,
    )
    from lane_measurements import (
        DEFAULT_LANE_WIDTH_METERS,
        DEFAULT_VISIBLE_ROAD_LENGTH_METERS,
        WIDTH_VARIATION_WARNING_PERCENT,
        calculate_measurements,
        draw_measurements,
        format_curvature,
        offset_direction,
    )
    from lane_overlay import draw_lane_overlay
    from perspective import perspective_transform
    from thresholding import combined_threshold, mask_for_saving
    from undistort import load_calibration, scale_camera_matrix, undistort_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
CALIBRATION_FILE = PROJECT_ROOT / "calibration_data" / "calibration.npz"

UNDISTORTION_OUTPUT_DIR = PROJECT_ROOT / "output_images" / "undistortion"
THRESHOLDING_OUTPUT_DIR = PROJECT_ROOT / "output_images" / "thresholding"
PERSPECTIVE_OUTPUT_DIR = PROJECT_ROOT / "output_images" / "perspective"
LANE_DETECTION_OUTPUT_DIR = PROJECT_ROOT / "output_images" / "lane_detection"
MEASUREMENTS_OUTPUT_DIR = PROJECT_ROOT / "output_images" / "lane_measurements"
OVERLAY_OUTPUT_DIR = PROJECT_ROOT / "output_images" / "lane_overlay"
FULL_PIPELINE_OUTPUT_DIR = PROJECT_ROOT / "output_images" / "full_pipeline"

CONTACT_ROWS_PER_SHEET = 4
NAME_CELL_WIDTH = 210
IMAGE_CELL_WIDTH = 320
IMAGE_CELL_HEIGHT = 180
HEADER_HEIGHT = 50


def process_image(image_path: Path, calibration) -> dict:
    """Run the shared pipeline once and return images and measurements."""

    original = cv2.imread(str(image_path))
    if original is None:
        raise RuntimeError("OpenCV could not decode the image.")

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

    left_x, left_y, right_x, right_y, windows = find_lane_pixels(warped)
    left_fit, right_fit, plot_y, left_fit_x, right_fit_x = (
        fit_lane_polynomials(
            left_x,
            left_y,
            right_x,
            right_y,
            warped.shape[0],
        )
    )

    sliding_windows = draw_sliding_windows(
        warped,
        windows,
        left_x,
        left_y,
        right_x,
        right_y,
    )
    polynomial_fit = draw_polynomial_fit(
        warped,
        left_x,
        left_y,
        right_x,
        right_y,
        plot_y,
        left_fit_x,
        right_fit_x,
    )

    width_diagnostics = lane_width_diagnostics(
        left_fit,
        right_fit,
        warped.shape[0],
    )
    (
        _,
        _,
        _,
        minimum_sampled_width,
        maximum_sampled_width,
        relative_width_variation,
    ) = width_diagnostics

    measurements = calculate_measurements(
        left_fit,
        right_fit,
        warped.shape[0],
        warped.shape[1],
        DEFAULT_LANE_WIDTH_METERS,
        DEFAULT_VISIBLE_ROAD_LENGTH_METERS,
    )
    measurements_image = draw_measurements(
        warped,
        left_x,
        left_y,
        right_x,
        right_y,
        plot_y,
        left_fit_x,
        right_fit_x,
        measurements,
    )
    overlay = draw_lane_overlay(
        undistorted,
        warped,
        inverse_matrix,
        plot_y,
        left_fit_x,
        right_fit_x,
    )

    stem = image_path.stem
    output_paths = {
        "undistorted": UNDISTORTION_OUTPUT_DIR / f"{stem}_undistorted.jpg",
        "binary": THRESHOLDING_OUTPUT_DIR / f"{stem}_binary.png",
        "warped": PERSPECTIVE_OUTPUT_DIR / f"{stem}_warped_binary.png",
        "sliding_windows": (
            LANE_DETECTION_OUTPUT_DIR / f"{stem}_sliding_windows.jpg"
        ),
        "polynomial_fit": (
            LANE_DETECTION_OUTPUT_DIR / f"{stem}_polynomial_fit.jpg"
        ),
        "measurements": MEASUREMENTS_OUTPUT_DIR / f"{stem}_measurements.jpg",
        "overlay": OVERLAY_OUTPUT_DIR / f"{stem}_lane_overlay.jpg",
    }
    write_image(output_paths["undistorted"], undistorted)
    write_image(output_paths["binary"], mask_for_saving(binary))
    write_image(output_paths["warped"], mask_for_saving(warped))
    write_image(output_paths["sliding_windows"], sliding_windows)
    write_image(output_paths["polynomial_fit"], polynomial_fit)
    write_image(output_paths["measurements"], measurements_image)
    write_image(output_paths["overlay"], overlay)

    return {
        "original": original,
        "polynomial_fit": polynomial_fit,
        "measurements_image": measurements_image,
        "overlay": overlay,
        "measurements": measurements,
        "minimum_sampled_width": float(minimum_sampled_width),
        "maximum_sampled_width": float(maximum_sampled_width),
        "relative_width_variation": float(relative_width_variation),
        "geometry_warning": (
            relative_width_variation > WIDTH_VARIATION_WARNING_PERCENT
        ),
        "output_paths": output_paths,
    }


def contact_image(image: np.ndarray) -> np.ndarray:
    """Convert a diagnostic result to a fixed-size BGR cell."""

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.resize(
        image,
        (IMAGE_CELL_WIDTH, IMAGE_CELL_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def create_contact_row(filename: str, result: dict) -> np.ndarray:
    name_cell = np.full(
        (IMAGE_CELL_HEIGHT, NAME_CELL_WIDTH, 3),
        32,
        dtype=np.uint8,
    )
    cv2.putText(
        name_cell,
        filename,
        (10, IMAGE_CELL_HEIGHT // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    images = (
        result["original"],
        result["polynomial_fit"],
        result["measurements_image"],
        result["overlay"],
    )
    return cv2.hconcat([name_cell, *[contact_image(image) for image in images]])


def create_header() -> np.ndarray:
    total_width = NAME_CELL_WIDTH + 4 * IMAGE_CELL_WIDTH
    header = np.full((HEADER_HEIGHT, total_width, 3), 32, dtype=np.uint8)
    labels = (
        ("Filename", 10),
        ("Original", NAME_CELL_WIDTH + 10),
        ("Polynomial fit", NAME_CELL_WIDTH + IMAGE_CELL_WIDTH + 10),
        ("Measurements", NAME_CELL_WIDTH + 2 * IMAGE_CELL_WIDTH + 10),
        ("Lane overlay", NAME_CELL_WIDTH + 3 * IMAGE_CELL_WIDTH + 10),
    )
    for label, x_position in labels:
        cv2.putText(
            header,
            label,
            (x_position, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return header


def save_contact_sheets(rows: list[np.ndarray]) -> list[Path]:
    """Save groups of full-pipeline results and return their paths."""

    paths = []
    header = create_header()
    for start in range(0, len(rows), CONTACT_ROWS_PER_SHEET):
        page_rows = rows[start : start + CONTACT_ROWS_PER_SHEET]
        sheet = cv2.vconcat([header, *page_rows])
        path = FULL_PIPELINE_OUTPUT_DIR / f"contact_sheet_{len(paths) + 1:02d}.jpg"
        write_image(path, sheet)
        paths.append(path)
    return paths


def report_lines(filename: str, result: dict) -> list[str]:
    measurements = result["measurements"]
    direction = offset_direction(measurements["offset_meters"])
    lines = [
        filename,
        f"  Left curvature: {format_curvature(measurements['left_curvature'])}",
        f"  Right curvature: {format_curvature(measurements['right_curvature'])}",
        f"  Vehicle offset: {measurements['offset_meters']:+.3f} m",
        f"  Vehicle position: {direction}",
        f"  Sampled lane-width range: "
        f"{result['minimum_sampled_width']:.2f}-"
        f"{result['maximum_sampled_width']:.2f} pixels",
        f"  Sampled lane-width variation: "
        f"{result['relative_width_variation']:.2f}%",
        f"  Geometry warning: "
        f"{'lane-width variation exceeds 10%' if result['geometry_warning'] else 'none'}",
    ]
    return lines


def main() -> int:
    image_paths = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    if not image_paths:
        print(f"Error: no JPG images found in {TEST_IMAGES_DIR}.", file=sys.stderr)
        return 1

    try:
        calibration = load_calibration(CALIBRATION_FILE)
    except (RuntimeError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    contact_rows = []
    successful_reports = []
    failures = []

    for image_path in image_paths:
        try:
            result = process_image(image_path, calibration)
            contact_rows.append(create_contact_row(image_path.name, result))
            image_report = report_lines(image_path.name, result)
            successful_reports.extend(image_report + [""])
            print(f"Processed: {image_path.name}")
            for line in image_report[1:]:
                print(line)
        except (
            RuntimeError,
            OSError,
            ValueError,
            OverflowError,
            cv2.error,
        ) as error:
            reason = str(error)
            failures.append((image_path.name, reason))
            print(f"Failed: {image_path.name}: {reason}", file=sys.stderr)

    contact_sheet_error = None
    try:
        contact_sheet_paths = save_contact_sheets(contact_rows)
    except (RuntimeError, OSError, cv2.error) as error:
        contact_sheet_paths = []
        contact_sheet_error = str(error)
        print(f"Contact-sheet error: {error}", file=sys.stderr)

    summary_lines = [
        "Full pipeline batch summary",
        f"Total images found: {len(image_paths)}",
        f"Successfully processed images: {len(contact_rows)}",
        f"Failed images: {len(failures)}",
        "",
        *successful_reports,
    ]
    if failures:
        summary_lines.append("Failures")
        summary_lines.extend(
            f"  {filename}: {reason}" for filename, reason in failures
        )
        summary_lines.append("")
    if contact_sheet_error is not None:
        summary_lines.append(f"Contact-sheet failure: {contact_sheet_error}")

    report_path = FULL_PIPELINE_OUTPUT_DIR / "full_pipeline_report.txt"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(summary_lines), encoding="utf-8")
    except OSError as error:
        print(f"Could not save batch report: {error}", file=sys.stderr)
        return 1

    print("\nBatch summary")
    print(f"Total images found: {len(image_paths)}")
    print(f"Successfully processed images: {len(contact_rows)}")
    print(f"Failed images: {len(failures)}")
    for filename, reason in failures:
        print(f"- {filename}: {reason}")
    print("Contact sheets:")
    if contact_sheet_paths:
        for path in contact_sheet_paths:
            print(f"- {path}")
    else:
        print("- none generated")
    print(f"Report: {report_path}")

    return 1 if failures or contact_sheet_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
