# Advanced Lane Finding

This university project develops an image-processing pipeline for identifying lane boundaries from a vehicle-mounted camera. The project is implemented incrementally. The current implementation contains Step 1, camera calibration, and Step 2, distortion correction for road images.

## Current project structure

```text
lane-finding/
├── camera_cal/                 # Original chessboard calibration photographs
├── examples/                   # Original project examples
├── src/
│   ├── calibration.py          # Step 1 calibration program
│   └── undistort.py            # Step 2 road-image distortion correction
├── test_images/                # Original test photographs
├── test_videos/                # Original test videos
├── .gitignore
├── README.md
├── requirements.txt
├── writeup.md                  # Project report in progress
└── writeup_template.md         # Original report template
```

The programs create `calibration_data/`, `output_images/calibration/`, and `output_images/undistortion/` when they are run. These generated directories are intentionally excluded from Git.

## Dependencies

- Python 3.10 or newer
- NumPy
- OpenCV for Python

The exact accepted package ranges are listed in `requirements.txt`.

## Environment setup on Ubuntu

From a terminal:

```bash
cd ~/lane-finding
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

Run `python src/calibration.py --help` to see all path and pattern options.

## Step 2: Correct distortion in a road image

Distortion correction requires the previously generated `calibration_data/calibration.npz` file and a road image captured at the saved calibration resolution. By default, the program reads `test_images/test1.jpg`:

```bash
python src/undistort.py
```

To process a different image, provide its path relative to the project root or as an absolute path:

```bash
python src/undistort.py --input-image test_images/test2.jpg
```

Optional `--calibration-file` and `--output-dir` arguments select different parameter and output locations. Run `python src/undistort.py --help` for the complete interface. The input image resolution must match the resolution stored in the calibration file; the program reports a clear error rather than applying incompatible parameters.

## Generated files

- `calibration_data/calibration.npz` stores the camera matrix, distortion coefficients, rotation and translation vectors, RMS reprojection error, image size, chessboard pattern size, and names of successful images.
- `output_images/calibration/corners/calibration*_corners.jpg` shows the refined corner detections for every successful calibration photograph; `corners_example.jpg` is a stable report-ready copy of the first successful detection.
- `output_images/calibration/undistorted_example.jpg` is an undistorted calibration photograph.
- `output_images/calibration/original_vs_undistorted.jpg` places the labelled original and undistorted photographs side by side.
- `output_images/undistortion/test1_undistorted.jpg` is the corrected default road image.
- `output_images/undistortion/test1_comparison.jpg` places the labelled original and corrected road images side by side. When a different input is selected, `test1` is replaced by that input file's stem.

The terminal output reports the number of readable calibration images processed, the number of successful detections, the camera matrix, distortion coefficients, RMS reprojection error, and saved-file locations. Images that cannot be read, have a different resolution, or do not contain the complete requested pattern are skipped with a warning.

## Check the calibration result

1. Confirm that the successful-detection count includes most of the readable input photographs. A few unsuccessful photographs can be normal if the entire pattern is not visible.
2. Inspect every generated corner image. The colored overlay should follow the actual internal chessboard intersections in the correct order.
3. Open the side-by-side comparison. Straight chessboard edges should look straighter after undistortion, particularly near the image boundaries, without implausible warping.
4. Confirm that `calibration.npz` was created and that the printed camera matrix and distortion coefficients contain finite values.
5. Record the printed detection counts and RMS reprojection error in `writeup.md`. A lower RMS error generally indicates better agreement, but visual inspection and a diverse set of board positions are also important.

## Check road-image distortion correction

1. Confirm that the program reports the same image resolution as the saved calibration.
2. Open `test1_undistorted.jpg` and confirm that it is complete and free from unexpected artifacts.
3. Inspect `test1_comparison.jpg`, concentrating on straight structures and lane markings near the image edges, where lens correction is usually most visible.
4. Confirm that the original file in `test_images/` remains unchanged.

Step 2 has not yet been executed or tested. Its output images and visual result must be checked during the manual verification step.
