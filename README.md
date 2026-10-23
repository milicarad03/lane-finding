# Advanced Lane Finding

This university project develops an image-processing pipeline for identifying lane boundaries from a vehicle-mounted camera. The project is implemented incrementally. The current implementation includes camera calibration, distortion correction, image thresholding, bird's-eye perspective transformation, and lane-pixel detection with polynomial fitting (Steps 1–5). Lane curvature, vehicle offset, lane overlay, and video processing are not yet implemented.

## Current project structure

```text
lane-finding/
├── camera_cal/                 # Original chessboard calibration photographs
├── calibration_data/
│   └── calibration.npz         # Saved camera calibration parameters
├── examples/                   # Original project examples
├── src/
│   ├── calibration.py          # Step 1 calibration program
│   ├── undistort.py            # Step 2 road-image distortion correction
│   ├── thresholding.py          # Step 3 color and gradient thresholding
│   ├── perspective.py           # Step 4 bird's-eye perspective transform
│   ├── lane_detection.py        # Step 5 sliding windows and polynomial fitting
│   └── process_all_images.py    # Batch processing and contact sheets (Steps 2–5)
├── output_images/              # Final demonstration results for Steps 1–5
│   ├── calibration/
│   ├── undistortion/
│   ├── thresholding/
│   ├── perspective/
│   ├── lane_detection/
│   └── batch/
├── test_images/                # Original test photographs
├── test_videos/                # Original test videos
├── .gitignore
├── README.md
├── requirements.txt
├── writeup.md                  # Project report in progress
└── writeup_template.md         # Original report template
```

The programs create `calibration_data/` and step-specific directories under `output_images/` when they are run. The cleaned `output_images/` directory is included in the repository so all final demonstration results for Steps 1–5 can be inspected. Its subdirectories organize calibration, undistortion, thresholding, perspective-transformation, lane-detection, and batch-processing results, including contact sheets. `writeup.md` uses representative examples rather than embedding every generated image.

## Dependencies

- Python 3.10 or newer
- NumPy
- OpenCV for Python

The exact accepted package ranges are listed in `requirements.txt`.

## Environment setup on Ubuntu

From a terminal:

```bash
cd lane-finding
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Step 1: Run camera calibration

From the project root, with the virtual environment active:

```bash
python src/calibration.py
```

Visual inspection of the supplied boards shows 10 by 7 squares, corresponding to the default 9 by 6 internal corners expected by OpenCV. If another calibration set is used, the internal-corner dimensions are configurable:

```bash
python src/calibration.py --corners-x 9 --corners-y 6
```

Run `python src/calibration.py --help` to see the configurable corner-dimension options.

## Step 2: Correct distortion in a road image

Distortion correction requires the previously generated `calibration_data/calibration.npz` file. The reusable `undistort_image()` function applies `cv2.undistort()` directly using the calibrated camera matrix and distortion coefficients, without `cv2.getOptimalNewCameraMatrix()`. The calibration example in `src/calibration.py` reuses the same function. By default, the program reads `test_images/test1.jpg`:

```bash
python src/undistort.py
```

To process a different image, provide its path relative to the project root or as an absolute path:

```bash
python src/undistort.py --input-image test_images/test2.jpg
```

Run `python src/undistort.py --help` to view the input-image option. The calibration and output locations remain fixed at `calibration_data/calibration.npz` and `output_images/undistortion/`. When an input has the calibration aspect ratio but a different resolution, the program scales `fx` and `cx` horizontally and `fy` and `cy` vertically before undistortion. This supports proportional sizes such as 1280×720 and 960×540. It assumes the image came from the same calibrated camera and was only resized; an incompatible aspect ratio produces an error.

## Step 3: Create a thresholded binary image

Image thresholding first undistorts the road image using the saved camera parameters. It then detects bright white markings and saturated yellow markings in HLS color space, calculates an X-direction Sobel gradient, and combines the three contributions into a binary lane-marking mask.

From the project root, run:

```bash
python src/thresholding.py
```

The default input is `test_images/test1.jpg`. To select another image:

```bash
python src/thresholding.py --input-image test_images/test2.jpg
```

The default thresholds are white HLS lightness 200–255 and saturation 0–80; yellow hue 15–35, lightness 30–255, and saturation 80–255; Sobel X magnitude 20–100 with a kernel size of 3; and a minimum lightness of 120 for gradient-only pixels. These values are grouped in `ThresholdConfig` so they can be adjusted in one place. Run `python src/thresholding.py --help` to view the input-image option.

## Step 4: Create a bird's-eye view

The perspective transform maps a trapezoidal lane region in the undistorted road image to a rectangle. `cv2.getPerspectiveTransform()` calculates the forward matrix `M` and inverse matrix `Minv`; `cv2.warpPerspective()` then transforms the combined binary image directly using nearest-neighbor interpolation. No additional polygon ROI mask is applied: comparison tests showed that this mask clipped some useful lane-marking pixels. The inverse matrix will later allow the detected lane boundaries to be projected back onto the road image. On straight-road examples, the transformation makes the lane boundaries appear approximately parallel and prepares the binary image for later lane-pixel and curvature calculations.

Run the default image from the project root:

```bash
python src/perspective.py
```

To select another image:

```bash
python src/perspective.py --input-image test_images/straight_lines1.jpg
```

The source points are stored as image-size ratios and were selected using the undistorted `straight_lines1.jpg`. Warped binary results were manually inspected on `test1.jpg`, `straight_lines1.jpg`, `straight_lines2.jpg`, and `solidYellowCurve.jpg`. Both boundaries are visible and approximately parallel in the straight-road examples. The `test1.jpg` result contains substantial extra thresholding noise, and the curved-road result contains gaps and noise; geometric accuracy across all scenes has not been established.

## Step 5: Detect lane pixels and fit polynomial curves

`src/lane_detection.py` reuses the existing undistortion, HLS/Sobel thresholding, and **unmasked** perspective transformation. It first sums nonzero pixels in each column of the lower half of the warped binary image. The strongest histogram peaks in the left and right halves provide starting positions for the lane boundaries.

A bottom-up sliding-window search uses **nine windows** for each boundary. The default horizontal search margin is **70 pixels at an image width of 1280 pixels**, scaled proportionally for other widths. Windows are recentered when enough lane-like pixels are available, allowing the search to follow solid or dashed markings. The selected pixel coordinates are independently fitted to second-degree polynomials using `np.polyfit(y, x, 2)`, representing `x(y) = A*y² + B*y + C` for the left and right boundaries. The implementation checks for insufficient usable pixels, insufficient vertical coverage, and crossing fitted curves; a successful fit by itself is not proof of lane-detection accuracy.

Run the default example (`test_images/straight_lines1.jpg`) from the project root:

```bash
python src/lane_detection.py
```

To process another photograph or reproduce the original wider-window experiment:

```bash
python src/lane_detection.py --input-image test_images/test1.jpg
python src/lane_detection.py --input-image test_images/test1.jpg --margin 100
```

An optional **experimental** `--robust-refinement` flag fits an initial polynomial, filters unusually large horizontal residuals using a median-absolute-deviation-based threshold, and refits the retained pixels. It is **not enabled by default**, because it did not show a clear improvement in the manual `test1.jpg` comparison.

```bash
python src/lane_detection.py --input-image test_images/test1.jpg --robust-refinement
python src/lane_detection.py --input-image test_images/test1.jpg --margin 70 --robust-refinement
```

The program saves the warped binary image, a visualization of the nine sliding windows and selected lane pixels, and a visualization of the fitted polynomials. It prints the left/right pixel counts and polynomial coefficients. The additional **lane-width diagnostic** evaluates the horizontal distance between the fitted curves at 20%, 50%, 80%, and the bottom of image height, then reports the minimum, maximum, and relative sampled-width variation. These values are **pixels in the transformed image**, not meters, and are diagnostic observations rather than automatic rejection criteria.

Manual inspection with the default **70-pixel margin** found identifiable left/right boundaries on `straight_lines1.jpg`, `straight_lines2.jpg`, `test1.jpg`, and `solidYellowCurve.jpg`. Sampled-width variations were **2.00%** for `straight_lines1.jpg`, **7.52%** for `test1.jpg`, and **11.81%** for `solidYellowCurve.jpg`. The last two cases warrant caution: `test1.jpg` includes unrelated selected pixels, and the 960×540 `solidYellowCurve.jpg` result also depends on the unverified same-camera-resize assumption. These tests do not establish physical or geometric accuracy on every image.

## Generated files

- `calibration_data/calibration.npz` stores the camera matrix, distortion coefficients, rotation and translation vectors, RMS reprojection error, image size, chessboard pattern size, and names of successful images.
- `output_images/calibration/corners/calibration*_corners.jpg` shows the refined corner detections for every successful calibration photograph; `corners_example.jpg` is a stable report-ready copy of the first successful detection.
- `output_images/calibration/undistorted_example.jpg` is an undistorted calibration photograph.
- `output_images/calibration/original_vs_undistorted.jpg` places the labelled original and undistorted photographs side by side.
- `output_images/undistortion/test1_undistorted.jpg` is the corrected default road image.
- `output_images/undistortion/test1_comparison.jpg` places the labelled original and corrected road images side by side. When a different input is selected, `test1` is replaced by that input file's stem.
- `output_images/thresholding/test1_white_mask.png` shows pixels selected as white lane markings.
- `output_images/thresholding/test1_yellow_mask.png` shows pixels selected as yellow lane markings.
- `output_images/thresholding/test1_sobel_mask.png` shows pixels selected by the X-direction gradient.
- `output_images/thresholding/test1_binary.png` is the final combined 0/255 binary image.
- `output_images/thresholding/test1_comparison.jpg` compares the original, undistorted, and combined binary images. With another input, `test1` is replaced by that file's stem.
- `output_images/perspective/test1_source_points.jpg` shows the proposed source quadrilateral on the undistorted road image.
- `output_images/perspective/test1_warped_binary.png` is the transformed bird's-eye binary image.
- `output_images/perspective/test1_warped_road.jpg` shows the undistorted color road transformed with the same perspective matrix.
- `output_images/perspective/test1_comparison.jpg` compares the thresholded binary image with its bird's-eye view. With another input, `test1` is replaced by that file's stem.
- `output_images/lane_detection/straight_lines1_warped_binary.png` shows the binary input used by Step 5 (using the default photograph).
- `output_images/lane_detection/straight_lines1_sliding_windows.jpg` shows sliding windows and selected left/right lane pixels.
- `output_images/lane_detection/straight_lines1_polynomial_fit.jpg` shows the left/right selected pixels and fitted polynomial curves. For other inputs, the stem changes; experimental configurations add suffixes such as `_margin100_` or `_robust_`.

The terminal output reports the number of readable calibration images processed, the number of successful detections, the camera matrix, distortion coefficients, RMS reprojection error, and saved-file locations. Images that cannot be read, have a different resolution, or do not contain the complete requested pattern are skipped with a warning.

## Check the calibration result

1. Confirm that the successful-detection count includes most of the readable input photographs. A few unsuccessful photographs can be normal if the entire pattern is not visible.
2. Inspect every generated corner image. The colored overlay should follow the actual internal chessboard intersections in the correct order.
3. Open the side-by-side comparison. Straight chessboard edges should look straighter after undistortion, particularly near the image boundaries, without implausible warping.
4. Confirm that `calibration.npz` was created and that the printed camera matrix and distortion coefficients contain finite values.
5. Record the printed detection counts and RMS reprojection error in `writeup.md`. A lower RMS error generally indicates better agreement, but visual inspection and a diverse set of board positions are also important.

## Check road-image distortion correction

1. Check the input image dimensions. If they differ from the calibration resolution, verify that the image was proportionally resized from the same calibrated camera.
2. Open `test1_undistorted.jpg` and confirm that it is complete and free from unexpected artifacts.
3. Inspect `test1_comparison.jpg`, concentrating on straight structures and lane markings near the image edges, where lens correction is usually most visible.
4. Confirm that the original file in `test_images/` remains unchanged.

## Check image thresholding

1. Inspect the white and yellow masks separately and confirm that each selects the intended lane-marking color.
2. Check that the Sobel mask emphasizes mostly vertical or diagonal lane edges without overwhelming background texture.
3. Inspect the final binary mask for continuous lane evidence on both sides of the vehicle and limited noise in the road surface, sky, vegetation, and vehicles.
4. Use the labelled comparison to confirm that selected pixels align with the visible lane markings in the undistorted image.
5. If necessary, adjust the threshold values in ThresholdConfig and rerun the program.

Steps 1–3 were manually inspected on representative road and calibration images after the distortion-correction change. Step 4 was visually inspected on `straight_lines1.jpg`, `straight_lines2.jpg`, `test1.jpg`, and `solidYellowCurve.jpg`. The batch pipeline processed all 17 images, including 960×540 inputs, without reported errors; that does not verify correct camera calibration or perspective geometry for every image. Intrinsic-matrix scaling is appropriate only when the image comes from the calibrated camera and was subsequently resized.

## Check the perspective transform

1. Inspect the source-point image and confirm that the quadrilateral follows the lane boundaries on the road surface.
2. Inspect the warped binary image and confirm that straight lane boundaries appear approximately vertical and parallel.
3. Check that the transform retains useful lane pixels without placing unrelated road regions inside the main warped lane area.
4. Adjust the clearly defined point ratios in `src/perspective.py` if the initial geometry is not suitable across the test images.

Some thresholding noise can remain, and expanding the source trapezoid can introduce visible stretching or blurring. Perspective geometry must therefore be inspected for each substantially different road view before later lane measurements are trusted.

## Check lane detection and polynomial fitting

1. Open `<stem>_sliding_windows.jpg` and confirm that the windows track the **two boundaries of the vehicle's own lane**, not unrelated road markings or noise.
2. Open `<stem>_polynomial_fit.jpg` and verify that both fitted curves follow the selected marking pixels, including the gaps in dashed lines.
3. Read the terminal pixel counts and polynomial coefficients; visually check the fit instead of equating more/fewer pixels with better quality.
4. Compare the lane-width values at different image heights. Large variations can suggest imperfect detection or perspective geometry, but the numbers are diagnostics, **not** evidence of a physically calibrated lane width.
5. For challenging scenes, compare `--margin 70` with `--margin 100` and, if useful, the optional `--robust-refinement` result. Experimental output names differ so the default images are preserved.

## Batch Processing

`process_all_images.py` loads the existing `calibration_data/calibration.npz` once, finds every JPG image in `test_images/`, and applies distortion correction, combined thresholding, the perspective transform, and the standard Step 5 lane detection without recalculating camera calibration. Lane detection uses the default 70-pixel sliding-window margin and the standard polynomial fit without optional robust refinement.

Run it from the project root:

```bash
python src/process_all_images.py
```

For every successfully processed input, the script saves the original, undistorted, combined binary, bird's-eye binary, sliding-window, and polynomial-fit images in `output_images/batch/`. It also creates numbered contact sheets with one row per successfully processed photograph. A failure is reported with its reason without stopping the remaining images. The batch pipeline successfully processed all 17 test images, and representative straight, curved, and challenging examples were visually inspected. The remaining images were not all individually validated, so successful processing does not imply perfect lane detection on every image.
