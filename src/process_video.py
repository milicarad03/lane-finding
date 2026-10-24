"""Process a road video through the complete lane-finding pipeline.

Run this module from the project root:

    python src/process_video.py
    python src/process_video.py --input-video test_videos/project_video01.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from .final_output import draw_final_output
    from .lane_detection import find_lane_pixels, fit_lane_polynomials
    from .lane_measurements import (
        DEFAULT_LANE_WIDTH_METERS,
        DEFAULT_VISIBLE_ROAD_LENGTH_METERS,
        calculate_measurements,
    )
    from .lane_overlay import draw_lane_overlay
    from .perspective import perspective_transform
    from .thresholding import combined_threshold
    from .undistort import load_calibration, scale_camera_matrix, undistort_image
except ImportError:
    from final_output import draw_final_output
    from lane_detection import find_lane_pixels, fit_lane_polynomials
    from lane_measurements import (
        DEFAULT_LANE_WIDTH_METERS,
        DEFAULT_VISIBLE_ROAD_LENGTH_METERS,
        calculate_measurements,
    )
    from lane_overlay import draw_lane_overlay
    from perspective import perspective_transform
    from thresholding import combined_threshold
    from undistort import load_calibration, scale_camera_matrix, undistort_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = PROJECT_ROOT / "calibration_data" / "calibration.npz"
DEFAULT_INPUT_VIDEO = PROJECT_ROOT / "test_videos" / "project_video01.mp4"
DEFAULT_OUTPUT_VIDEO = (
    PROJECT_ROOT / "output_videos" / "project_video_output.mp4"
)
PROGRESS_INTERVAL_FRAMES = 50
TRACKING_MARGIN_AT_1280 = 100
FIT_SMOOTHING_FACTOR = 0.25
MAXIMUM_REUSED_FIT_SECONDS = 2.0
MAXIMUM_CENTER_SHIFT_AT_1280 = 120.0
MINIMUM_WIDTH_RATIO = 0.50
MAXIMUM_WIDTH_RATIO = 1.50
REINITIALIZE_AFTER_REUSED_FRAMES = 5
MINIMUM_PLAUSIBLE_WIDTH_RATIO = 0.08
MAXIMUM_PLAUSIBLE_WIDTH_RATIO = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a video through the complete lane-finding pipeline."
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        default=DEFAULT_INPUT_VIDEO,
        help="video to process (default: test_videos/project_video01.mp4)",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=DEFAULT_OUTPUT_VIDEO,
        help=(
            "processed video path "
            "(default: output_videos/project_video_output.mp4)"
        ),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    """Resolve a relative command-line path from the project root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def search_around_polynomials(
    binary_image: np.ndarray,
    left_fit: np.ndarray,
    right_fit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find lane pixels close to the preceding frame's fitted curves."""

    height, width = binary_image.shape
    margin = max(
        1,
        int(round(TRACKING_MARGIN_AT_1280 * width / 1280)),
    )
    nonzero_y, nonzero_x = binary_image.nonzero()
    left_distance = np.abs(nonzero_x - np.polyval(left_fit, nonzero_y))
    right_distance = np.abs(nonzero_x - np.polyval(right_fit, nonzero_y))

    # Assign a pixel to only one boundary when the two search bands overlap.
    left_selected = (left_distance <= margin) & (
        left_distance < right_distance
    )
    right_selected = (right_distance <= margin) & (
        right_distance < left_distance
    )
    return (
        nonzero_x[left_selected],
        nonzero_y[left_selected],
        nonzero_x[right_selected],
        nonzero_y[right_selected],
    )


def evaluate_polynomials(
    left_fit: np.ndarray,
    right_fit: np.ndarray,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate finite, non-crossing lane curves for every image row."""

    plot_y = np.arange(image_height, dtype=np.float64)
    left_fit_x = np.polyval(left_fit, plot_y)
    right_fit_x = np.polyval(right_fit, plot_y)
    if not (
        np.all(np.isfinite(left_fit))
        and np.all(np.isfinite(right_fit))
        and np.all(np.isfinite(left_fit_x))
        and np.all(np.isfinite(right_fit_x))
    ):
        raise RuntimeError("The tracked lane fit contains invalid values.")
    if np.any(left_fit_x >= right_fit_x):
        raise RuntimeError("The tracked left and right lane curves cross.")
    return plot_y, left_fit_x, right_fit_x


class VideoLaneTracker:
    """Track, validate, and smooth lane fits between consecutive frames."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        maximum_reused_frames: int,
    ) -> None:
        self.image_height = image_height
        self.image_width = image_width
        self.maximum_reused_frames = maximum_reused_frames
        self.left_fit = None
        self.right_fit = None
        self.consecutive_reused_frames = 0
        self.maximum_consecutive_reused_frames = 0
        self.tracking_frames = 0
        self.sliding_window_frames = 0
        self.reused_frames = 0
        self.reinitialized_frames = 0

    def validate_absolute_geometry(
        self,
        candidate_left: np.ndarray,
        candidate_right: np.ndarray,
    ) -> None:
        """Require plausible positive lane width independent of prior frames."""

        bottom_y = self.image_height - 1
        sample_y = np.array(
            [
                round(0.20 * bottom_y),
                round(0.50 * bottom_y),
                round(0.80 * bottom_y),
                bottom_y,
            ],
            dtype=np.float64,
        )
        widths = (
            np.polyval(candidate_right, sample_y)
            - np.polyval(candidate_left, sample_y)
        )
        minimum_width = MINIMUM_PLAUSIBLE_WIDTH_RATIO * self.image_width
        maximum_width = MAXIMUM_PLAUSIBLE_WIDTH_RATIO * self.image_width
        if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
            raise RuntimeError("The sampled lane width is invalid.")
        if np.any(widths < minimum_width) or np.any(widths > maximum_width):
            raise RuntimeError(
                "The sampled lane width is outside the plausible image range."
            )

    def validate_temporal_change(
        self,
        candidate_left: np.ndarray,
        candidate_right: np.ndarray,
    ) -> None:
        """Reject a sudden lane-center or lane-width jump between frames."""

        if self.left_fit is None or self.right_fit is None:
            return

        bottom_y = self.image_height - 1
        sample_y = np.array(
            [
                round(0.20 * bottom_y),
                round(0.50 * bottom_y),
                round(0.80 * bottom_y),
                bottom_y,
            ],
            dtype=np.float64,
        )
        previous_left = np.polyval(self.left_fit, sample_y)
        previous_right = np.polyval(self.right_fit, sample_y)
        candidate_left_x = np.polyval(candidate_left, sample_y)
        candidate_right_x = np.polyval(candidate_right, sample_y)
        previous_width = previous_right - previous_left
        candidate_width = candidate_right_x - candidate_left_x

        if np.any(previous_width <= 0) or np.any(candidate_width <= 0):
            raise RuntimeError("The sampled tracked lane width is not positive.")

        previous_center = (previous_left + previous_right) / 2.0
        candidate_center = (candidate_left_x + candidate_right_x) / 2.0
        maximum_center_shift = (
            MAXIMUM_CENTER_SHIFT_AT_1280 * self.image_width / 1280
        )
        if np.max(np.abs(candidate_center - previous_center)) > maximum_center_shift:
            raise RuntimeError(
                "The detected lane center changed too abruptly from the "
                "preceding frame."
            )

        width_ratio = candidate_width / previous_width
        if np.any(width_ratio < MINIMUM_WIDTH_RATIO) or np.any(
            width_ratio > MAXIMUM_WIDTH_RATIO
        ):
            raise RuntimeError(
                "The detected lane width changed too abruptly from the "
                "preceding frame."
            )

    def accept_fit(
        self,
        candidate_left: np.ndarray,
        candidate_right: np.ndarray,
        compare_to_previous: bool = True,
        smooth_with_previous: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validate and smooth a newly detected polynomial pair."""

        evaluate_polynomials(
            candidate_left,
            candidate_right,
            self.image_height,
        )
        self.validate_absolute_geometry(candidate_left, candidate_right)
        if compare_to_previous:
            self.validate_temporal_change(candidate_left, candidate_right)

        if (
            self.left_fit is None
            or self.right_fit is None
            or not smooth_with_previous
        ):
            smoothed_left = candidate_left
            smoothed_right = candidate_right
        else:
            smoothed_left = (
                (1.0 - FIT_SMOOTHING_FACTOR) * self.left_fit
                + FIT_SMOOTHING_FACTOR * candidate_left
            )
            smoothed_right = (
                (1.0 - FIT_SMOOTHING_FACTOR) * self.right_fit
                + FIT_SMOOTHING_FACTOR * candidate_right
            )

        plot_y, left_fit_x, right_fit_x = evaluate_polynomials(
            smoothed_left,
            smoothed_right,
            self.image_height,
        )
        self.left_fit = smoothed_left
        self.right_fit = smoothed_right
        self.consecutive_reused_frames = 0
        return (
            smoothed_left,
            smoothed_right,
            plot_y,
            left_fit_x,
            right_fit_x,
        )

    def fit(
        self,
        warped_binary: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        str,
        str | None,
    ]:
        """Fit one frame, preferring tracked pixels over a full new search."""

        errors = []
        if self.left_fit is not None and self.right_fit is not None:
            try:
                left_x, left_y, right_x, right_y = search_around_polynomials(
                    warped_binary,
                    self.left_fit,
                    self.right_fit,
                )
                candidate_left, candidate_right, _, _, _ = (
                    fit_lane_polynomials(
                        left_x,
                        left_y,
                        right_x,
                        right_y,
                        self.image_height,
                    )
                )
                accepted = self.accept_fit(candidate_left, candidate_right)
                self.tracking_frames += 1
                return (*accepted, "tracking", None)
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                errors.append(f"tracked search: {error}")

        reinitialization_candidate = None
        try:
            left_x, left_y, right_x, right_y, _ = find_lane_pixels(
                warped_binary
            )
            candidate_left, candidate_right, _, _, _ = fit_lane_polynomials(
                left_x,
                left_y,
                right_x,
                right_y,
                self.image_height,
            )
            reinitialization_candidate = (candidate_left, candidate_right)
            accepted = self.accept_fit(candidate_left, candidate_right)
            self.sliding_window_frames += 1
            return (*accepted, "sliding windows", None)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            sliding_error = error
            errors.append(f"sliding windows: {sliding_error}")

        if (
            self.left_fit is not None
            and self.right_fit is not None
            and self.consecutive_reused_frames
            >= REINITIALIZE_AFTER_REUSED_FRAMES - 1
            and reinitialization_candidate is not None
        ):
            candidate_left, candidate_right = reinitialization_candidate
            try:
                accepted = self.accept_fit(
                    candidate_left,
                    candidate_right,
                    compare_to_previous=False,
                    smooth_with_previous=False,
                )
                self.sliding_window_frames += 1
                self.reinitialized_frames += 1
                return (
                    *accepted,
                    "sliding-window reinitialization",
                    str(sliding_error),
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                errors.append(f"reinitialization: {error}")

        if self.left_fit is None or self.right_fit is None:
            raise RuntimeError("; ".join(errors))
        if self.consecutive_reused_frames >= self.maximum_reused_frames:
            raise RuntimeError(
                "No reliable current-frame lane fit was available for more than "
                f"{self.maximum_reused_frames} consecutive frames. "
                + "; ".join(errors)
            )

        self.consecutive_reused_frames += 1
        self.reused_frames += 1
        self.maximum_consecutive_reused_frames = max(
            self.maximum_consecutive_reused_frames,
            self.consecutive_reused_frames,
        )
        plot_y, left_fit_x, right_fit_x = evaluate_polynomials(
            self.left_fit,
            self.right_fit,
            self.image_height,
        )
        return (
            self.left_fit,
            self.right_fit,
            plot_y,
            left_fit_x,
            right_fit_x,
            "reused previous fit",
            "; ".join(errors),
        )


def add_tracking_warning(image: np.ndarray, text: str) -> np.ndarray:
    """Add a visible video-tracking status warning to one frame."""

    result = image.copy()
    position = (40, 195)
    cv2.putText(
        result,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def process_frame(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    tracker: VideoLaneTracker,
) -> tuple[np.ndarray, str, str | None]:
    """Apply Steps 2-8 to one BGR video frame."""

    if frame is None or frame.size == 0:
        raise RuntimeError("Cannot process an empty video frame.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError("Video frames must contain three BGR channels.")

    undistorted = undistort_image(
        frame,
        camera_matrix,
        distortion_coefficients,
    )
    binary = combined_threshold(undistorted)
    warped, _, inverse_matrix, _, _ = perspective_transform(binary)

    (
        left_fit,
        right_fit,
        plot_y,
        left_fit_x,
        right_fit_x,
        fit_source,
        fit_warning,
    ) = tracker.fit(warped)
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
    final_frame = draw_final_output(overlay, measurements)
    if fit_source == "reused previous fit":
        final_frame = add_tracking_warning(
            final_frame,
            "Warning: previous valid lane fit temporarily reused",
        )
    elif fit_source == "sliding-window reinitialization":
        final_frame = add_tracking_warning(
            final_frame,
            "Warning: lane tracker reinitialized from current frame",
        )
    return final_frame, fit_source, fit_warning


def main() -> int:
    args = parse_args()
    input_path = project_path(args.input_video)
    output_path = project_path(args.output_video)

    capture = None
    writer = None
    processed_frames = 0

    try:
        if not input_path.is_file():
            raise RuntimeError(f"Input video not found: {input_path}")
        if input_path.resolve() == output_path.resolve():
            raise RuntimeError("Input and output video paths must be different.")

        calibration = load_calibration(CALIBRATION_FILE)
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open the video: {input_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise RuntimeError("The input video reports an invalid resolution.")
        if not np.isfinite(fps) or fps <= 0:
            raise RuntimeError("The input video reports an invalid frame rate.")

        camera_matrix = scale_camera_matrix(
            calibration.camera_matrix,
            calibration.image_size,
            (width, height),
        )
        maximum_reused_frames = max(
            1,
            int(round(MAXIMUM_REUSED_FIT_SECONDS * fps)),
        )
        tracker = VideoLaneTracker(
            height,
            width,
            maximum_reused_frames,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            codec,
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV could not create the video: {output_path}")

        while True:
            frame_available, frame = capture.read()
            if not frame_available:
                break

            frame_number = processed_frames + 1
            if frame.shape[:2] != (height, width):
                raise RuntimeError(
                    f"Frame {frame_number} has resolution "
                    f"{frame.shape[1]}x{frame.shape[0]}, expected {width}x{height}."
                )

            try:
                final_frame, fit_source, fit_warning = process_frame(
                    frame,
                    camera_matrix,
                    calibration.distortion_coefficients,
                    tracker,
                )
            except (
                RuntimeError,
                OSError,
                ValueError,
                OverflowError,
                cv2.error,
            ) as error:
                raise RuntimeError(
                    f"Lane detection failed on frame {frame_number}: {error}"
                ) from error

            if fit_source == "reused previous fit":
                print(
                    f"Warning: frame {frame_number} reused the preceding valid "
                    f"lane fit ({fit_warning}).",
                    file=sys.stderr,
                )
            elif fit_source == "sliding-window reinitialization":
                print(
                    f"Warning: frame {frame_number} reinitialized the lane "
                    f"tracker ({fit_warning}).",
                    file=sys.stderr,
                )
            writer.write(final_frame)
            processed_frames = frame_number
            if (
                processed_frames == 1
                or processed_frames % PROGRESS_INTERVAL_FRAMES == 0
            ):
                if expected_frames > 0:
                    print(f"Processed frame {processed_frames}/{expected_frames}")
                else:
                    print(f"Processed frame {processed_frames}")

        if processed_frames == 0:
            raise RuntimeError("The input video did not contain a readable frame.")
        if expected_frames > 0 and processed_frames < expected_frames:
            raise RuntimeError(
                f"Video decoding stopped after {processed_frames} of "
                f"{expected_frames} reported frames."
            )
    except (RuntimeError, OSError, ValueError, OverflowError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()

    print("\nVideo-processing summary")
    print(f"Input video: {input_path}")
    print(f"Output video: {output_path}")
    print(f"Resolution: {width}x{height}")
    print(f"Frame rate: {fps:.3f} FPS")
    print(f"Processed frames: {processed_frames}")
    print(f"Fits from tracked search: {tracker.tracking_frames}")
    print(f"Fits from sliding windows: {tracker.sliding_window_frames}")
    print(f"Tracker reinitializations: {tracker.reinitialized_frames}")
    print(f"Frames reusing the previous fit: {tracker.reused_frames}")
    print(
        "Maximum consecutive reused fits: "
        f"{tracker.maximum_consecutive_reused_frames}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
