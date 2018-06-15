## Writeup
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/vehicle_non_vehicle.png
[image2]: ./output_images/hls_vehicle_hog.png
[image3]: ./output_images/pipeline_detection.png
[image4]: ./output_images/pipeline_detection2.png
[video1]: ./output_videos/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cells 2-6 of the IPython notebook `Vehicle_Detection.ipynb`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are some random samples from each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I used the random samples from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like, for each of the channels separately and the whole image as-is.

Here is an example using the `HLS` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

From the HOG visualizations mentioned above, I found that the L channel in HLS/LUV space, any of the RGB channels, Y channel in YCrCb space, and S channel in the HSV space all produce visually good HOG gradients. I compared all these side-by-side, and didn't find any major differences. I also visualized various combinations of HOG parameters, some of which are shown in the table below. In general, 8 or 9 orientations, with 8x8 pixels/cell produced good HOG visualizations. Different values of cells/block didn't seem to make any difference.

| Color Space | Orientations | Pix / cell | Cells / block | HOG channels |
|:-----------:|:------------:|:----------:|:-------------:|:------------:|
|     RGB     |       9      |      8     |       2       | All, 0, 1, 2 |  
|     HSV     |       9      |      8     |       2       | All, 0, 1, 2 |
|    YCrCb    |       9      |      8     |       2       | All, 0, 1, 2 |
|     LUV     |       9      |      8     |       2       | All, 0, 1, 2 |
|     HLS     |       9      |      8     |       2       | All, 0, 1, 2 |
|     HLS     |       9      |      4     |       2       | All, 0, 1, 2 |
|     HLS     |       8      |      8     |       2       | All, 0, 1, 2 |
|     HLS     |       8      |      4     |       4       | All, 0, 1, 2 |
|     HLS     |       8      |      16    |       2       | All, 0, 1, 2 |
|     HLS     |       8      |      4     |       4       | All, 0, 1, 2 |

Thus I decided to start with orientations = 8, pixels/cell = 8x8, and cells/block = 2.
See code cell 10. I visualized the spatial bins of various color spaces and found that the RGB channel gives the best spatial bin results (clear difference between vehicle and non-vehicle images).
I also visualized color histogram features for each channel in various color spaces and found that the `L` channel in HLS space & `Y` channel in the YCrCb space seems to give the best results.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

See the code cells 11-14 in the IPynb notebook.
I extracted features using function `extract_features()`, which uses `single_img_features()` to extract features from a single image. For feature extraction, I started with the values for various HOG, spatial bin, and histogram parameters that I thought would work best based on the visualizations mentioned above. I generated the y-labels from the lists `vehicles` (1) and `non_vehicles` (0). Then I scaled the features using `sklearn.prerpocessing.StandardScaler()`. I shuffled the combined data set & split it into training and test data sets using `sklearn.model_selection.train_test_split()`
I trained a linear SVM using `LinearSVC()` from the sklearn library.

Here are some of the combinations I tried. Initially, I kept the color space same for the entire pipeline (spatial, histogram, HOG), but later also experimented with using different color spaces for spatial vs. histogram vs. HOG features, and separate channels for histogram and HOG.

| ClrSpace | Orient. | Pix/cell | Cells/blk | Spat.size | Hist.bins | Train time | Test Acc |
|:--------:|:-------:|:--------:|:---------:|:---------:|:---------:|:----------:|:---------|
|     L    |    9    |    8     |     2     |   16x16   |    16     |   6.01s    |  98.73%  |
|     L    |    9    |    4     |     2     |   16x16   |    16     |  11.55s    |  98.28%  |
|    HLS   |    9    |    8     |     2     |   16x16   |    16     |   2.97s    |  98.79%  |
|     Y    |    9    |    8     |     2     |   16x16   |    16     |   5.64s    |  98.45%  |
|   YCrCb  |    9    |    8     |     2     |   16x16   |    16     |   2.56s    |  99.21%  |
|   YCrCb  |    9    |    8     |     2     |   32x32   |    16     |  20.59s    |  99.41%  |
|   YCrCb  |    9    |    8     |     2     |   32x32   |    32     |  20.71s    |  99.30%  |
|   YCrCb  |    8    |    8     |     2     |   32x32   |    16     |  17.18s    |  98.96%  |
|   YCrCb  |    8    |    8     |     2     |   16x16   |    16     |  11.41s    |  99.52%  |
|    HSV   |    9    |    8     |     2     |   32x32   |    16     |  20.71s    |  99.38%  |
|    RGB   |    9    |    8     |     2     |   32x32   |    16     |  29.91s    |  98.56%  |

I also tried non-linear SVC(), more sophisticated scaling using QuantileTransformer() from Scikit-learn library, but that did not help in improving the accuracy much, and was much slower.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

See code cells 15-16. I use the logic from the FAQ video and tweaked it a little. The function `draw_boxes()` draws boxes on given image using given box boundaries. The function `draw_labeled_bboxes()` is used to draw boxes using labeled heatmaps. The function `find_vehicle_boxes()` uses the given `y_start_stop` boundaries to crop the image, scale it using the given `scale`, and slides a window of size 64x64 with the provided `overlap` value. As smaller vehicles are likely to be found farther from the bottom (near the top portion of the cropped image), I adjust the `y_stop` value accordingly. I extract the same pipeline of spatial bin, histogram, and HOG features as the one used during training, with the same classifier to predict vehicle positions, and generate bounding boxes based on that. For each positive prediction, a heatmap is incremented for the pixels within the bounding box.
I tried various combinations of scales and overlap values and found that scales = [1, 1.5, 2.5], [1.25, 1.5, 2.5], or [1.5, 2, 2.5] and overlap = 75% give reasonably good results (visually) for the set of test images. However, each combination results into some false positives and some false negatives. I further experimented with various threshold and scale values to eliminate as many false negatives / false positives as possible.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As mentioned above, I experimented with many different combinations of color spaces, channels, and HOG, spatial bin parameters until I found the combination that consistently performs better than others.
Ultimately I settled on 3 scales - 1.25, 1.5, 2.5 using YCrCb 3-channel HOG features plus spatially binned color and Y-channel color histograms in the feature vector, which provided a nice result. The classifier accuracy with this combination is always above 99%.
Here is another example image:

![alt text][image4]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the bounding boxes of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual areas in the heatmap.  I then assumed each area corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

A problem I saw was there was always a trade-off between test dataset accuracy and the speed of training as well as predicting accurate bounding boxes. I added several code improvements such as adjusting the cropped image size based on the scale, using only 1 channel for histogram, etc. to reduce the size of the feature vector.

Another problem is the trade-off between overlap, multiple scale values and detection speed: higher overlap & many different values of scale ensures better vehicle detection, but at the cost of the speed of detection.

Also, the pipeline fails to remove all false positives and false negatives. It is necessary to further experiment with more combinations of scales, thresholds, and overlap values. Also, extracting more vehicle and non-vehicle samples from the project video and adding them to the training data set will help reduce the errors.  
