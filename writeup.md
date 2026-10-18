## Writeup Template

### You use this file as a template for your writeup.

---

**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
[//]: # (Image References)

[calibration-corners]: ./output_images/calibration/corners/corners_example.jpg "Detected chessboard corners"
[calibration-comparison]: ./output_images/calibration/original_vs_undistorted.jpg "Original and undistorted calibration image"

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera-calibration implementation is in `src/calibration.py`. Visual inspection of the supplied calibration photographs shows 10 squares horizontally and 7 squares vertically, which produces 9 by 6 internal chessboard corners. These counts are the program defaults and can be changed with `--corners-x` and `--corners-y` rather than being fixed inside the detection logic.

Each calibration photograph is loaded with OpenCV and converted from BGR to grayscale. `cv2.findChessboardCorners()` searches for the complete internal-corner pattern. When detection succeeds, `cv2.cornerSubPix()` refines the corner positions to sub-pixel accuracy. Failed detections are reported and skipped without adding an invalid correspondence. A copy of each successful image is saved with the refined corners drawn by `cv2.drawChessboardCorners()`.

![Detected and refined chessboard corners][calibration-corners]

The corresponding three-dimensional object points describe a planar board at `z = 0`. Their coordinates are `(0, 0, 0)`, `(1, 0, 0)`, and so on over the 9 by 6 grid. The physical square size is therefore treated as one arbitrary unit; that is sufficient for estimating the intrinsic camera matrix and lens distortion. The same object-point grid is paired with the refined two-dimensional image points from every successful photograph.

`cv2.calibrateCamera()` uses those object/image-point correspondences and the photograph dimensions to calculate the camera matrix and distortion coefficients. It also returns the rotation and translation vectors for each successful view and the RMS reprojection error. The program saves all these values and useful metadata in `calibration_data/calibration.npz`.

The implementation uses `cv2.getOptimalNewCameraMatrix()` followed by `cv2.undistort()` to correct one successful calibration photograph. It writes the corrected image separately and also creates a labelled, side-by-side comparison with the original.

![Original and undistorted calibration image][calibration-comparison]

The calibration program examined 20 input images, of which 15 were successfully used for camera calibration. Three images—`calibration1.jpg`, `calibration4.jpg`, and `calibration5.jpg`—were skipped because `cv2.findChessboardCorners()` could not detect the complete pattern. Two more images—`calibration7.jpg` and `calibration15.jpg`—were skipped because their resolution was 1281 by 721 pixels rather than the expected 1280 by 720 pixels. Images with inconsistent dimensions cannot safely be combined in one calibration because `cv2.calibrateCamera()` expects all collected image points to use the same image coordinate system.

The calculated camera matrix was:

```text
[[1.15877475e+03 0.00000000e+00 6.69642741e+02]
 [0.00000000e+00 1.15407661e+03 3.88079450e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
```

The calculated distortion coefficients were:

```text
[[-2.56779082e-01  4.33845159e-02 -6.87454487e-04
   1.25769022e-04 -1.15025457e-01]]
```

The RMS reprojection error was **0.852880 pixels**. Manual inspection of the generated diagnostic images confirmed that the detected and refined corners were correctly positioned on the internal chessboard intersections. The original-versus-undistorted comparison also showed reduced lens distortion after correction, providing a successful visual verification of the calibration and undistortion results.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The implementation for this step is located in `src/undistort.py`. It loads the camera matrix, distortion coefficients, and calibration image size from `calibration_data/calibration.npz`; it does not hardcode or recalculate the camera parameters. The input image resolution is checked against the saved calibration resolution before correction so incompatible parameters are not applied silently.

The reusable `undistort_image()` function passes an already-loaded image together with the saved camera matrix and distortion coefficients to `cv2.undistort()`. Keeping this operation independent of file input/output allows later stages of the lane-finding pipeline to apply the same correction to their image frames.

Correcting lens distortion before lane detection is necessary because radial and tangential distortion can bend straight lane markings and shift their apparent pixel positions, especially near the image boundaries. Removing that distortion gives the later perspective transformation and lane-geometry calculations a more accurate image.

For the default `test_images/test1.jpg` input, the corrected road image is written to `output_images/undistortion/test1_undistorted.jpg`. A labelled side-by-side comparison of the original and corrected images is written to `output_images/undistortion/test1_comparison.jpg`.

![Original and undistorted road image][road-undistortion-comparison]

[road-undistortion-comparison]: ./output_images/undistortion/test1_comparison.jpg "Original and undistorted road image"

Manual testing was completed by running `python src/undistort.py`. The program successfully processed `test_images/test1.jpg` at its 1280 by 720 pixel resolution and generated `output_images/undistortion/test1_undistorted.jpg` and `output_images/undistortion/test1_comparison.jpg`. Visual inspection confirmed that the corrected road image looks natural, the lane markings remain clearly visible, and there are no obvious distortion artifacts.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients, or other methods to create a thresholded binary image. Provide an example of a binary image result.

TODO — future project step.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

TODO — future project step.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial.

TODO — future project step.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TODO — future project step.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

TODO — future project step.

---

### Pipeline (video)

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video.

TODO — future project step.

---

### Discussion

#### 1. Briefly discuss any problems or issues faced in the implementation. Where will the pipeline likely fail, and what could make it more robust?

TODO — complete after the full lane-finding pipeline has been implemented and evaluated.
