"""Apply a bird's-eye perspective transform to a thresholded road image.

Run this module from the project root:

    python src/perspective.py
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from .thresholding import combined_threshold
    from .undistort import load_calibration, scale_camera_matrix, undistort_image
except ImportError:
    from thresholding import combined_threshold
    from undistort import load_calibration, scale_camera_matrix, undistort_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = PROJECT_ROOT / "calibration_data" / "calibration.npz"
DEFAULT_INPUT_IMAGE = PROJECT_ROOT / "test_images" / "test1.jpg"
OUTPUT_DIR = PROJECT_ROOT / "output_images" / "perspective"

# Initial points based on the visible lane boundaries in undistorted
# 1280x720 straight_lines1.jpg.
# They are ratios so the same layout scales to proportional image sizes.
SOURCE_POINT_RATIOS = np.float32(
    [
        [585 / 1280, 460 / 720],  # top-left
        [705 / 1280, 460 / 720],  # top-right
        [1095 / 1280, 700 / 720], # bottom-right
        [240 / 1280, 700 / 720],  # bottom-left
    ]
)
DESTINATION_POINT_RATIOS = np.float32(
    [
        [320 / 1280, 0.0],       # top-left
        [960 / 1280, 0.0],       # top-right
        [960 / 1280, 719 / 720], # bottom-right
        [320 / 1280, 719 / 720], # bottom-left
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a bird's-eye view of a thresholded road image."
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        default=DEFAULT_INPUT_IMAGE,
        help="road image to process (default: test_images/test1.jpg)",
    )
    return parser.parse_args()


def perspective_points(image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return source and destination points for the given width and height."""

    width, height = image_size
    scale = np.float32([width, height])
    source_points = SOURCE_POINT_RATIOS * scale
    destination_points = DESTINATION_POINT_RATIOS * scale
    return source_points.astype(np.float32), destination_points.astype(np.float32)


def perspective_transform(binary_image: np.ndarray):
    """Warp a 0/1 binary image and return the forward and inverse matrices."""

    if binary_image is None or binary_image.size == 0 or binary_image.ndim != 2:
        raise RuntimeError("Perspective transform requires a non-empty binary image.")

    height, width = binary_image.shape
    source_points, destination_points = perspective_points((width, height))
    matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    inverse_matrix = cv2.getPerspectiveTransform(destination_points, source_points)

    warped = cv2.warpPerspective(
        binary_image,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
    )
    warped = (warped > 0).astype(np.uint8)
    return warped, matrix, inverse_matrix, source_points, destination_points


def draw_source_points(image: np.ndarray, source_points: np.ndarray) -> np.ndarray:
    """Draw the source quadrilateral and its ordered corner labels."""

    marked = image.copy()
    height, width = marked.shape[:2]
    display_points = np.round(source_points).astype(np.int32)
    display_points[:, 0] = np.clip(display_points[:, 0], 0, width - 1)
    display_points[:, 1] = np.clip(display_points[:, 1], 0, height - 1)
    cv2.polylines(
        marked,
        [display_points.reshape(-1, 1, 2)],
        isClosed=True,
        color=(0, 0, 255),
        thickness=4,
        lineType=cv2.LINE_AA,
    )

    labels = ("TL", "TR", "BR", "BL")
    for label, (x, y) in zip(labels, display_points):
        cv2.circle(marked, (int(x), int(y)), 8, (0, 255, 0), -1)
        cv2.putText(
            marked,
            label,
            (int(x) + 10, max(int(y) - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return marked


def mask_for_saving(mask: np.ndarray) -> np.ndarray:
    return (mask * 255).astype(np.uint8)


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
        calibration = load_calibration(CALIBRATION_FILE)
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
        undistorted = undistort_image(
            original,
            camera_matrix,
            calibration.distortion_coefficients,
        )
        binary = combined_threshold(undistorted)
        (
            warped,
            matrix,
            inverse_matrix,
            source_points,
            destination_points,
        ) = perspective_transform(binary)

        output_paths = {
            "source": OUTPUT_DIR / f"{input_path.stem}_source_points.jpg",
            "warped": OUTPUT_DIR / f"{input_path.stem}_warped_binary.png",
            "warped_road": OUTPUT_DIR / f"{input_path.stem}_warped_road.jpg",
            "comparison": OUTPUT_DIR / f"{input_path.stem}_comparison.jpg",
        }
        write_image(
            output_paths["source"],
            draw_source_points(undistorted, source_points),
        )

        binary_image = mask_for_saving(binary)
        warped_image = mask_for_saving(warped)
        write_image(output_paths["warped"], warped_image)
        warped_road = cv2.warpPerspective(
            undistorted,
            matrix,
            (undistorted.shape[1], undistorted.shape[0]),
            flags=cv2.INTER_LINEAR,
        )
        write_image(output_paths["warped_road"], warped_road)
        comparison = cv2.hconcat(
            [
                add_label(
                    cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR),
                    "Thresholded binary",
                ),
                add_label(
                    cv2.cvtColor(warped_image, cv2.COLOR_GRAY2BGR),
                    "Bird's-eye binary",
                ),
            ]
        )
        write_image(output_paths["comparison"], comparison)
    except (RuntimeError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print("Source points:")
    print(source_points)
    print("Destination points:")
    print(destination_points)
    print("Perspective matrix M:")
    print(matrix)
    print("Inverse perspective matrix Minv:")
    print(inverse_matrix)
    for output_path in output_paths.values():
        print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
