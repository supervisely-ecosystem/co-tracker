<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/co-tracker/assets/119248312/8349a1da-6f97-4063-a731-826a1d758d66" />
  
# CoTracker object tracking

<p align="center">
  <a href="#Original-work">Original work</a> •
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Demo">Demo</a> 
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/co-tracker/supervisely_integration/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/co-tracker)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/co-tracker/supervisely_integration/serve.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/co-tracker/supervisely_integration/serve.png)](https://supervise.ly)

</div>

# Original work

Original work is available here: [**paper**](https://arXiv:2307.07635); [**code**](https://github.com/facebookresearch/co-tracker); [**project**](https://co-tracker.github.io/)

«This architecture is based on several ideas from the optical flow and tracking literature, and combines them in a new, flexible and powerful design. It is based on a transformer network that models the correlation of different points in time via specialised attention layers.

The transformer is designed to update iteratively an estimate of several trajectories. It can be applied in a sliding-window manner to very long videos, for which we engineer an unrolled training loop. It compares favourably against state-of-the-art point tracking methods, both in terms of efficiency and accuracy»

<img src="https://github.com/supervisely-ecosystem/co-tracker/assets/119248312/0710ae69-3140-42e4-a3d0-f3cddf08bfa1" />

### Points on a uniform grid

> We track points sampled on a regular grid starting from the initial video frame. The colors represent the object (magenta) and the background (cyan).

https://user-images.githubusercontent.com/119248312/275542057-d4cbc02e-fd74-4492-b0c0-6dc476df1677.mp4

### Individual points

> We track the same queried point with different methods and visualize its trajectory using color encoding based on time. The red cross (❌) indicates the ground truth point coordinates.

https://user-images.githubusercontent.com/119248312/275542284-01cdf2d4-4816-4ec5-a8ec-fe4610839792.mp4

# Overview

This app is an integration of CoTracker model, which is a NN-assisted interactive object tracking model. CoTracker is a fast transformer-based model that can track any point in a video. It brings to tracking some of the benefits of Optical Flow.

#### This application allows you to track the following 4 types of figures:

- Point

- Line

- Keypoints

- Polygon

# How to Run

0. Go to Ecosystem page and find the [app](https://ecosystem.supervisely.com/apps/co-tracker/supervisely_integration/serve). 

1. Select one of the suggested checkpoints.

<img src="https://github.com/supervisely-ecosystem/co-tracker/assets/119248312/a1f3ea25-eedb-42ea-bb6c-12272b2bc350" width="500" />

2. Run the app on an agent with `CPU` or `GPU`. For **Community Edition** - users have to run the app on their own GPU computer connected to the platform. Watch this [video tutorial](https://youtu.be/aO7Zc4kTrVg).

2. Open [video labeling](https://app.supervisely.com/ecosystem/annotation_tools/video-labeling-tool?id=178) interface.

3. Configure tracking settings.

4. Press `Track` button.

5. When you have finished working with the application, manually stop the app session in the `App Sessions` tab.

https://user-images.githubusercontent.com/119248312/275678239-96a47023-6344-48f3-96cf-84ab98aeeb47.mp4


### You can also use this app to track keypoints. This app can track keypoint graphs of any shape and number of points.

1. Open your video project, select a suitable frame and click the `Screenshot` button in the top right corner.

2. Create keypoints class based on your screenshot.

3. Go back to the video, set your recently created keypoint graph on the target object, select number of frames to track and press `Track` button.

**You can change the visualisation settings of your keypoint graph at any time in the right sidebar**

https://user-images.githubusercontent.com/119248312/275678304-8a8ddcc6-547e-4539-a84f-62431c9e7988.mp4

# Demo

### Keypoints

https://user-images.githubusercontent.com/119248312/275669028-58f263b7-29b9-4923-91ac-7df1419d827d.mp4

### Line

https://user-images.githubusercontent.com/119248312/275678424-3389cdd4-af13-4040-9909-51ac47874f67.mp4

### Point

https://user-images.githubusercontent.com/119248312/275678477-a7924c40-4c61-48db-8b38-9dd50cb51a90.mp4
