"""Process every JPG test image through Steps 2-5 of the pipeline.

Run this module from the project root:

    python src/process_all_images.py
"""

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
    )
    from .perspective import perspective_transform
    from .thresholding import combined_threshold
    from .undistort import load_calibration, scale_camera_matrix, undistort_image
except ImportError:
    from lane_detection import (
        draw_polynomial_fit,
        draw_sliding_windows,
        find_lane_pixels,
        fit_lane_polynomials,
    )
    from perspective import perspective_transform
    from thresholding import combined_threshold
    from undistort import load_calibration, scale_camera_matrix, undistort_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
CALIBRATION_FILE = PROJECT_ROOT / "calibration_data" / "calibration.npz"
OUTPUT_DIR = PROJECT_ROOT / "output_images" / "batch"

CONTACT_ROWS_PER_SHEET = 4
NAME_CELL_WIDTH = 200
IMAGE_CELL_WIDTH = 320
IMAGE_CELL_HEIGHT = 180
HEADER_HEIGHT = 50


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def process_image(image_path: Path, calibration):
    """Process one image using the already-loaded calibration data."""

    original = cv2.imread(str(image_path))
    if original is None:
        raise RuntimeError("OpenCV could not decode the image.")

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
    binary = combined_threshold(undistorted)
    warped, _, _, _, _ = perspective_transform(binary)
    left_x, left_y, right_x, right_y, windows = find_lane_pixels(warped)
    _, _, plot_y, left_fit_x, right_fit_x = fit_lane_polynomials(
        left_x,
        left_y,
        right_x,
        right_y,
        warped.shape[0],
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

    stem = image_path.stem
    write_image(OUTPUT_DIR / f"{stem}_original.jpg", original)
    write_image(OUTPUT_DIR / f"{stem}_undistorted.jpg", undistorted)
    write_image(OUTPUT_DIR / f"{stem}_binary.png", binary * 255)
    write_image(OUTPUT_DIR / f"{stem}_warped_binary.png", warped * 255)
    write_image(OUTPUT_DIR / f"{stem}_sliding_windows.jpg", sliding_windows)
    write_image(OUTPUT_DIR / f"{stem}_polynomial_fit.jpg", polynomial_fit)
    return original, undistorted, binary, warped, sliding_windows, polynomial_fit


def contact_image(image: np.ndarray) -> np.ndarray:
    """Convert a result to a fixed-size BGR contact-sheet cell."""

    if image.ndim == 2:
        image = cv2.cvtColor(image * 255, cv2.COLOR_GRAY2BGR)
    return cv2.resize(
        image,
        (IMAGE_CELL_WIDTH, IMAGE_CELL_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def create_contact_row(filename: str, images) -> np.ndarray:
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
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return cv2.hconcat([name_cell, *[contact_image(image) for image in images]])


def create_header() -> np.ndarray:
    total_width = NAME_CELL_WIDTH + 6 * IMAGE_CELL_WIDTH
    header = np.full((HEADER_HEIGHT, total_width, 3), 32, dtype=np.uint8)
    labels = (
        ("Filename", 10),
        ("Original", NAME_CELL_WIDTH + 10),
        ("Undistorted", NAME_CELL_WIDTH + IMAGE_CELL_WIDTH + 10),
        ("Thresholded binary", NAME_CELL_WIDTH + 2 * IMAGE_CELL_WIDTH + 10),
        ("Bird's-eye binary", NAME_CELL_WIDTH + 3 * IMAGE_CELL_WIDTH + 10),
        ("Sliding windows", NAME_CELL_WIDTH + 4 * IMAGE_CELL_WIDTH + 10),
        ("Polynomial fit", NAME_CELL_WIDTH + 5 * IMAGE_CELL_WIDTH + 10),
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


def save_contact_sheets(rows) -> list[Path]:
    """Save readable groups of result rows and return their paths."""

    contact_sheet_paths = []
    header = create_header()
    for start in range(0, len(rows), CONTACT_ROWS_PER_SHEET):
        page_rows = rows[start : start + CONTACT_ROWS_PER_SHEET]
        contact_sheet = cv2.vconcat([header, *page_rows])
        page_number = len(contact_sheet_paths) + 1
        path = OUTPUT_DIR / f"contact_sheet_{page_number:02d}.jpg"
        write_image(path, contact_sheet)
        contact_sheet_paths.append(path)
    return contact_sheet_paths


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
    failures = []
    for image_path in image_paths:
        try:
            results = process_image(image_path, calibration)
            contact_rows.append(create_contact_row(image_path.name, results))
            print(f"Processed: {image_path.name}")
        except (RuntimeError, OSError, ValueError, cv2.error) as error:
            reason = str(error)
            failures.append((image_path.name, reason))
            print(f"Failed: {image_path.name}: {reason}", file=sys.stderr)

    contact_sheet_error = None
    try:
        contact_sheet_paths = save_contact_sheets(contact_rows)
    except (RuntimeError, OSError, cv2.error) as error:
        print(f"Error while creating contact sheets: {error}", file=sys.stderr)
        contact_sheet_paths = []
        contact_sheet_error = str(error)

    print("\nBatch summary")
    print(f"Total images found: {len(image_paths)}")
    print(f"Successfully processed images: {len(contact_rows)}")
    print(f"Failed images: {len(failures)}")
    for filename, reason in failures:
        print(f"- {filename}: {reason}")
    if contact_sheet_error is not None:
        print(f"Contact-sheet failure: {contact_sheet_error}")

    print("Contact sheets:")
    if contact_sheet_paths:
        for path in contact_sheet_paths:
            print(f"- {path}")
    else:
        print("- none generated")

    return 1 if failures or contact_sheet_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
