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

The calibration program reuses `undistort_image()` from `src/undistort.py` to correct one successful calibration photograph with the calibrated camera matrix and distortion coefficients via `cv2.undistort()`. It saves the corrected image and a labelled, side-by-side comparison. The comparison was regenerated and manually inspected after removing `cv2.getOptimalNewCameraMatrix()`; the chessboard geometry looked reasonable and the visible distortion was reduced.

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

The implementation for this step is located in `src/undistort.py`. It loads the camera matrix, distortion coefficients, and calibration image size from `calibration_data/calibration.npz`; it does not hardcode or recalculate the camera parameters. For an image at the original calibration resolution, the saved camera matrix is used unchanged. For a proportionally resized image, `scale_camera_matrix()` scales `fx` and `cx` by the width ratio and `fy` and `cy` by the height ratio while leaving the distortion coefficients unchanged. This assumes that the image came from the same calibrated camera and was only resized. An incompatible aspect ratio produces an error rather than silently applying incorrect parameters.

The reusable `undistort_image()` function uses `cv2.undistort()` directly with the saved camera matrix and distortion coefficients. It does not call `cv2.getOptimalNewCameraMatrix()`; for images at the calibration resolution, the calibrated matrix is also used as the output camera matrix. For images that were proportionally resized from the calibrated camera, `scale_camera_matrix()` first adjusts the intrinsic parameters. Keeping this operation independent of file input/output allows subsequent pipeline stages to apply the same correction to their image frames.

Correcting lens distortion before lane detection is necessary because radial and tangential distortion can bend straight lane markings and shift their apparent pixel positions, especially near the image boundaries. Removing that distortion gives the later perspective transformation and lane-geometry calculations a more accurate image.

For the default `test_images/test1.jpg` input, the corrected road image is written to `output_images/undistortion/test1_undistorted.jpg`. A labelled side-by-side comparison of the original and corrected images is written to `output_images/undistortion/test1_comparison.jpg`.

![Original and undistorted road image][road-undistortion-comparison]

[road-undistortion-comparison]: ./output_images/undistortion/test1_comparison.jpg "Original and undistorted road image"

Manual testing was completed by running `python src/undistort.py`. The program successfully processed `test_images/test1.jpg` at its 1280 by 720 pixel resolution and generated `output_images/undistortion/test1_undistorted.jpg` and `output_images/undistortion/test1_comparison.jpg`. Visual inspection confirmed that the corrected road image looks natural, the lane markings remain clearly visible, and there are no obvious distortion artifacts.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients, or other methods to create a thresholded binary image. Provide an example of a binary image result.

The image-thresholding implementation is located in `src/thresholding.py`. It first uses the saved camera parameters and the reusable distortion-correction function from `src/undistort.py`, then applies color and gradient thresholds to the undistorted BGR image. The processing functions operate on image arrays independently of file input/output so they can be reused in the complete lane-finding pipeline.

HLS color space separates hue and saturation from lightness, which makes the colors of lane markings easier to describe independently of much of the scene brightness. The white mask selects pixels with lightness from 200 to 255 and saturation from 0 to 80. The high lightness selects bright markings, while the low saturation distinguishes white from strongly colored pixels. The yellow mask uses OpenCV HLS hue values from 15 to 35, lightness from 30 to 255, and saturation from 80 to 255. These ranges select saturated yellow markings while rejecting most neutral road pixels. The threshold values are defined in ThresholdConfig and can be adjusted if necessary.

The Sobel mask is calculated from the HLS lightness channel. `cv2.Sobel()` computes the first derivative in the X direction with a kernel size of 3. The absolute gradient is safely normalized to the 0–255 range; when the maximum gradient is zero, the function returns an all-zero mask instead of dividing by zero. The default normalized-gradient range is 20 to 100.

The reusable `combined_threshold()` function returns a binary array containing only 0 and 1. It combines all white and yellow color detections with Sobel detections whose HLS lightness is at least 120. The additional lightness condition retains useful lane edges while reducing gradient responses from dark road texture. Saved masks are multiplied by 255 so standard image viewers display them correctly.

For the default input, the individual masks and final binary image are saved in `output_images/thresholding/`. The labelled comparison at `output_images/thresholding/test1_comparison.jpg` contains the original road image, the undistorted image, and the combined binary result.

![Original, undistorted, and thresholded road image][thresholding-comparison]

[thresholding-comparison]: ./output_images/thresholding/test1_comparison.jpg "Original, undistorted, and thresholded road image"

The thresholding stage was originally tested on the eight 1280 by 720 test images (`test1.jpg` through `test6.jpg`, `straight_lines1.jpg`, and `straight_lines2.jpg`). After the distortion-correction change, the binary results for `test1.jpg`, `straight_lines1.jpg`, and `straight_lines2.jpg` were regenerated and manually inspected: both lane boundaries remained identifiable, although some white markings were only partially selected. The masks also contained unwanted responses from vegetation, cars, guardrails, and road texture, especially from Sobel gradients. The final batch pipeline subsequently processed all 17 test images, including nine 960 by 540 photographs, with zero reported failures. Processing success does not establish that the same-camera assumption holds for every input or that all lane markings are accurately extracted.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective-transform implementation is located in `src/perspective.py`. A bird's-eye view is useful because lane boundaries that converge toward the horizon in the road image become approximately vertical and parallel after rectification. This geometry supports the sliding-window lane-pixel search and polynomial fitting implemented in Step 5.

The source points define a trapezoid around the expected lane boundaries in the undistorted road image. Their current 1280 by 720 coordinates are `(585, 460)`, `(705, 460)`, `(1015, 650)`, and `(310, 650)`, ordered top-left, top-right, bottom-right, and bottom-left. The destination points use the same order and form a rectangle at `(320, 0)`, `(960, 0)`, `(960, 719)`, and `(320, 719)`. In the code, both sets are stored as width/height ratios so they scale with proportionally resized images. The source quadrilateral follows the solid yellow and dashed white boundaries of the driving lane approximately, while the destination rectangle makes those boundaries approximately vertical and parallel.

`cv2.getPerspectiveTransform()` calculates the forward matrix `M` from the source and destination points. The inverse matrix `Minv` is calculated by reversing those point sets; it will later allow detected lane boundaries to be projected back onto the road image. `cv2.warpPerspective()` applies `M` directly to the combined 0/1 binary image using nearest-neighbor interpolation so the transformed mask remains binary. No additional polygon mask is applied: manual comparison of masked and unmasked results showed that the mask clipped useful lane-marking pixels. The same perspective matrix is also applied to the undistorted color road image for diagnostic inspection.

The source-point visualization, warped binary image, warped color road image, and comparison are saved in `output_images/perspective/`. The comparison for the default image is shown below.

![Thresholded and bird's-eye binary images][perspective-comparison]

[perspective-comparison]: ./output_images/perspective/test1_comparison.jpg "Thresholded and bird's-eye binary images"

The final unmasked perspective transformation was manually inspected on `straight_lines1.jpg`, `straight_lines2.jpg`, `test1.jpg`, and `solidYellowCurve.jpg`. The straight-road results retained both lane boundaries, which appeared approximately vertical and parallel. On `test1.jpg`, both lane markings were visible but substantial extra thresholding noise remained. The transformed `solidYellowCurve.jpg` image retained some recognizable lane-marking evidence, alongside considerable noise and gaps; its geometric accuracy was not established. Expanding the source trapezoid into a rectangle may produce stretching and interpolation artifacts, particularly in the diagnostic color image. The final batch script processed all 17 test images without reported errors, but accuracy and robustness across all road views—and the applicability of calibration to every input—have not yet been verified.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial.

The lane-pixel detection and polynomial-fitting implementation is in `src/lane_detection.py`. It reuses the calibrated distortion correction (`src/undistort.py`), the HLS/Sobel combined threshold (`src/thresholding.py`), and the **unmasked** bird's-eye transformation (`src/perspective.py`); camera calibration and the earlier image-processing stages are not repeated. The input to lane detection is the transformed 0/1 binary image.

**Finding lane pixels.** A column-wise histogram counts nonzero pixels in the **lower half** of the warped binary image. The strongest peak in each half provides an initial horizontal position for the left and right lane boundaries. Starting at the bottom, a **nine-level sliding-window search** collects nonzero pixel coordinates within windows around those estimates. The default horizontal margin is **70 pixels for a 1280-pixel-wide image** (scaled with image width), and windows recenter when enough candidate pixels are present. This allows the method to follow both solid and dashed markings, including windows with gaps between dashed segments. The original 100-pixel margin remains selectable with `--margin 100` for comparison.

**Fitting the lane boundaries.** The candidate left and right pixels are fitted separately using `np.polyfit(y, x, 2)` to obtain the second-degree polynomials `x_left(y) = A_l*y² + B_l*y + C_l` and `x_right(y) = A_r*y² + B_r*y + C_r`. The fitted curves estimate the horizontal positions of the lane boundaries at each image row. The program reports errors for insufficient pixel evidence or vertical coverage and checks whether the fitted curves cross; it does not fabricate missing boundaries. An optional, **non-default** `--robust-refinement` experiment rejects unusually large horizontal residuals using a median-absolute-deviation-based threshold before refitting. Manual comparison on `test1.jpg` did not demonstrate a clear visual advantage, so the ordinary fit remains the default.

**Visual examples of lane detection.** The following figures illustrate the nine-window search and second-degree polynomial fitting on different lane markings. The individually tested examples use the default **70-pixel margin** (at 1280-pixel image width). Blue and red indicate selected left/right lane pixels, and yellow and green show the fitted polynomials. Dashed-line gaps can be bridged by fitting a curve to visible segments. The two additional examples selected from the batch-image archive are presented as visual illustrations; the archive alone does not verify their runtime options or numerical accuracy.

**Example 1 — `straight_lines1.jpg`: solid left boundary and dashed right boundary.** The sliding windows follow both markings, and the fitted curves remain close to the selected pixels. The dashed right boundary is represented by a continuous polynomial even where no white marking is present.

![Sliding-window detection on straight_lines1.jpg][lane-sliding-windows]

![Fitted lane boundaries on straight_lines1.jpg][lane-polynomial-fit]

**Example 2 — `straight_lines2.jpg`: dashed left boundary and solid right boundary.** This scene checks the reverse marking configuration. Several left-hand windows contain no marking, but the detected dashed segments still provide enough evidence for a fitted curve, while the right-hand windows follow the solid white line.

![Sliding-window detection on straight_lines2.jpg][lane-sliding-windows-2]

![Fitted lane boundaries on straight_lines2.jpg][lane-polynomial-fit-2]

**Example 3 — `solidYellowCurve.jpg`: another road view at 960 × 540 pixels.** The program identifies visible portions of both boundaries and fits curves across the dashed sections. This demonstrates that the implementation executes at a second image resolution; it does **not** establish that the camera parameters or perspective geometry are physically accurate for this input.

![Fitted lane boundaries on solidYellowCurve.jpg][lane-polynomial-fit-curve]

**Example 4 — `test2.jpg`: a more pronounced road bend.** The original road image provides context for the curved yellow left boundary and dashed white right boundary. In the saved polynomial-fit visualization from the batch results, the selected pixels and fitted curves follow the visible markings through the bend. This is a qualitative example, not a validated physical curvature measurement.

![Original road image for test2.jpg][lane-test2-original]

![Fitted lane boundaries on test2.jpg][lane-test2-fit]

**Example 5 — `solidWhiteCurve.jpg`: dashed left boundary and solid right boundary on a curved road.** Unlike the straight-road examples, this image shows a bending road with a solid white line on the right. The batch-result visualization shows the fit through separated left-hand dashes and along the continuous right-hand marking. This 960 × 540 example is also subject to the previously stated camera-resizing and perspective-geometry assumptions.

![Original road image for solidWhiteCurve.jpg][lane-white-curve-original]

![Fitted lane boundaries on solidWhiteCurve.jpg][lane-white-curve-fit]

[lane-sliding-windows]: ./output_images/lane_detection/straight_lines1_sliding_windows.jpg "Sliding-window search on straight_lines1.jpg"
[lane-polynomial-fit]: ./output_images/lane_detection/straight_lines1_polynomial_fit.jpg "Polynomial fits on straight_lines1.jpg"
[lane-sliding-windows-2]: ./output_images/lane_detection/straight_lines2_sliding_windows.jpg "Sliding-window search on straight_lines2.jpg"
[lane-polynomial-fit-2]: ./output_images/lane_detection/straight_lines2_polynomial_fit.jpg "Polynomial fits on straight_lines2.jpg"
[lane-polynomial-fit-curve]: ./output_images/lane_detection/solidYellowCurve_polynomial_fit.jpg "Polynomial fits on solidYellowCurve.jpg"
[lane-test2-original]: ./output_images/batch/test2_original.jpg "Original image: test2.jpg"
[lane-test2-fit]: ./output_images/batch/test2_polynomial_fit.jpg "Polynomial fits from batch results: test2.jpg"
[lane-white-curve-original]: ./output_images/batch/solidWhiteCurve_original.jpg "Original image: solidWhiteCurve.jpg"
[lane-white-curve-fit]: ./output_images/batch/solidWhiteCurve_polynomial_fit.jpg "Polynomial fits from batch results: solidWhiteCurve.jpg"

**Testing and limitations.** The default 70-pixel margin was manually tested on `straight_lines1.jpg`, `straight_lines2.jpg`, `test1.jpg`, and `solidYellowCurve.jpg`. Both lane boundaries were identifiable in these examples. The program found **31,465 left / 5,139 right** candidate pixels on `straight_lines1.jpg`, **5,181 / 14,281** on `straight_lines2.jpg`, **49,331 / 6,078** on `test1.jpg`, and **15,010 / 4,794** on `solidYellowCurve.jpg`. The `test1.jpg` result still includes unrelated selected pixels around the left marking; detecting many pixels is not necessarily evidence of a better fit. On the 960×540 `solidYellowCurve.jpg` image, the resized-camera assumption also remains unverified.

A further diagnostic computes `x_right(y) - x_left(y)` at several image heights and reports the relative variation of the sampled lane width **in warped-image pixels**. The tested results were **2.00%** on `straight_lines1.jpg`, **5.76%** on `straight_lines2.jpg`, **7.52%** on `test1.jpg`, and **11.81%** on `solidYellowCurve.jpg`. These variations indicate how the *estimated* lane separation changes with height; they are not physical lane widths and are not hard-coded acceptance/rejection thresholds. They can reflect image noise, polynomial-fit error, or imperfect perspective geometry. The camera/perspective geometry has not been verified as metrically accurate across every test image. Curvature in meters and vehicle offset are intentionally left for the next step.


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
