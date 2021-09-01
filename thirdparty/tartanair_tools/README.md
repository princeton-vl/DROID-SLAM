# TartanAir dataset: AirSim Simulation Dataset for Simultaneous Localization and Mapping
This repository provides sample code and scripts for accessing the training and testing data, as well as evaluation tools. Please refer to [TartanAir](http://theairlab.org/tartanair-dataset) for more information about the dataset. 
You can also reach out to contributors on the associated [AirSim GitHub](https://github.com/microsoft/AirSim).

This dataset was used to train the first generalizable learning-based visual odometry [TartanVO](http://theairlab.org/tartanvo/), which achieved better performance than geometry-based VO methods in challenging cases. Please check out the [paper](https://arxiv.org/pdf/2011.00359.pdf) and published [TartanVO code](https://github.com/castacks/tartanvo). 

## Download the training data

The data is divided into two levels (Easy and Hard) in terms of the motion patterns. It is organized in trajectory folders. You can download data from different cameras (left or right), with different data types (RGB, depth, segmentation, camera pose, and flow). Please see [data type](data_type.md) page for the camera intrinsics, extrinsics and other information. 


<p style="color:red"> <b> !! NOTE: The size of all the data is up to 3TB! It could take days to download. We also added the option to use the dataset directly on Azure without requiring a download. Please select the data type you really need before download. You can also go to 
<a href=http://theairlab.org/tartanair-dataset>TartanAir</a> 
to download the sample files for a better understanding of the data types. </b> </p>

###  Data directory structure

```
ROOT
|
--- ENV_NAME_0                             # environment folder
|       |
|       ---- Easy                          # difficulty level
|       |      |
|       |      ---- P000                   # trajectory folder
|       |      |      |
|       |      |      +--- depth_left      # 000000_left_depth.npy - 000xxx_left_depth.npy
|       |      |      +--- depth_right     # 000000_right_depth.npy - 000xxx_right_depth.npy
|       |      |      +--- flow            # 000000_000001_flow/mask.npy - 000xxx_000xxx_flow/mask.npy
|       |      |      +--- image_left      # 000000_left.png - 000xxx_left.png 
|       |      |      +--- image_right     # 000000_right.png - 000xxx_right.png 
|       |      |      +--- seg_left        # 000000_left_seg.npy - 000xxx_left_seg.npy
|       |      |      +--- seg_right       # 000000_right_seg.npy - 000xxx_right_seg.npy
|       |      |      ---- pose_left.txt 
|       |      |      ---- pose_right.txt
|       |      |  
|       |      +--- P001
|       |      .
|       |      .
|       |      |
|       |      +--- P00K
|       |
|       +--- Hard
|
+-- ENV_NAME_1
.
.
|
+-- ENV_NAME_N
```

### Download data to your local machine

We provide a python script `download_training.py` for the data downloading. You can also take look at the [URL list](download_training_zipfiles.txt) to download the spesific files you want. 

* Specify an output directory

  --output-dir OUTPUTDIR

* Select file type:

  --rgb

  --depth

  --seg

  --flow

* Select difficulty level:
  
  --only-hard

  --only-easy

  [NO TAG]: both 'hard' and 'easy' levels

* Select camera:
  
  --only-left

  --only-right

  [NO TAG]: both 'left' and 'right' cameras

* Select flow type when --flow is set:
  
  --only-flow

  --only-mask

  [NO TAG]: both 'flow' and 'mask' files

For example, download all the RGB images from the left camera:

```
python download_training.py --output-dir OUTPUTDIR --rgb --only-left
```

Download all the depth data from both cameras in hard level: 

```
python download_training.py --output-dir OUTPUTDIR --depth --only-hard
```

Download all optical flow data without flow-mask:

```
python download_training.py --output-dir OUTPUTDIR --flow --only-flow
```

Download all the files in the dataset (could be very slow due to the large size):

```
python download_training.py --output-dir OUTPUTDIR --rgb --depth --seg --flow
```

---
**NOTE**

We found that using AzCopy, which is a tool provided by MicroSoft, is much faster than directly downloading by the URLs. Please try the following steps if you want to accelerate the downloading process. 

1. Download the [AzCopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10) and put the executable to your system path. 

2. Run the above commands with a --azcopy tag. 

---

### Access the data using Azure virtual machine

Yet another way to access the data is to use an Azure virtual machine. In this way, you don't have to download the data to your local machine. You can use [Azure SDKs](https://github.com/Azure/azure-sdk-for-python) to access the data on the cloud. [Here](TartanAir_Sample.ipynb) is a sample notebook for accessing and visualizing the data. **NOTE: This sample notebook can only be used on Azure. To download the data to your local machine, please refer to the download instructions [here](https://github.com/castacks/tartanair_tools#download-data-to-your-local-machine) or the [dataset website](http://theairlab.org/tartanair-dataset) for the sample data.**

## Download the testing data for the CVPR Visual SLAM challenge

You can click the download links below. In Linux system, you can also run the following command to download all the files: 
```
wget -r -i download_cvpr_slam_test.txt
``` 

* [Monocular track](https://tartanair.blob.core.windows.net/tartanair-testing1/tartanair-test-mono-release.tar.gz) (Size: 7.65 GB)
  
  MD5 hash: 009b52e7d7b224ffb8a203db294ac9fb

```
mono
|
--- ME000                             # monocular easy trajectory 0 
|       |
|       ---- 000000.png               # RGB image 000000
|       ---- 000001.png               # RGB image 000001
|       .
|       .
|       ---- 000xxx.png               # RGB image 000xxx
|
+-- ME001                             # monocular easy trajectory 1 
.
.
+-- ME007                             # monocular easy trajectory 7 
|
+-- MH000                             # monocular hard trajectory 0 
.
.
|
+-- MH007                             # monocular hard trajectory 7 
```

* [Stereo track](https://tartanair.blob.core.windows.net/tartanair-testing1/tartanair-test-stereo-release.tar.gz) (Size: 17.51 GB)

  MD5 hash: 8a3363ff2013f147c9495d5bb161c48e

```
stereo
|
--- SE000                                 # stereo easy trajectory 0 
|       |
|       ---- image_left                   # left image folder
|       |       |
|       |       ---- 000000_left.png      # RGB left image 000000
|       |       ---- 000001_left.png      # RGB left image 000001
|       |       .
|       |       .
|       |       ---- 000xxx_left.png      # RGB left image 000xxx
|       |
|       ---- image_right                  # right image folder
|               |
|               ---- 000000_right.png     # RGB right image 000000
|               ---- 000001_right.png     # RGB right image 000001
|               .
|               .
|               ---- 000xxx_right.png     # RGB right image 000xxx
|
+-- SE001                                 # stereo easy trajectory 1 
.
.
+-- SE007                                 # stereo easy trajectory 7 
|
+-- SH000                                 # stereo hard trajectory 0 
.
.
|
+-- SH007                                 # stereo hard trajectory 7 
```

* [Both monocular and stereo tracks](https://tartanair.blob.core.windows.net/tartanair-testing1/tartanair-test-release.tar.gz) (Size: 25.16 GB)

  MD5 hash: ea176ca274135622cbf897c8fa462012 

More information about the [CVPR Visual SLAM challenge](https://sites.google.com/view/vislocslamcvpr2020/slam-challenge)

* The [monocular track](https://www.aicrowd.com/challenges/tartanair-visual-slam-mono-track)

* The [stereo track](https://www.aicrowd.com/challenges/tartanair-visual-slam-stereo-track)

Now the CVPR challenge has completed, if you need the <b> ground truth poses </b> for the above testing trajectories, please send an email to [tartanair@hotmail.com](tartanair@hotmail.com). 

## Evaluation tools

Following the [TUM dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset) and the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php), we adopt three metrics: absolute trajectory error (ATE), the relative pose error (RPE), a modified version of KITTI VO metric. 

[More details](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#evaluation)

Check out the sample code: 
```
cd evaluation
python tartanair_evaluator.py
```

Note that our camera poses are defined in the NED frame. That is to say, the x-axis is pointing to the camera's forward, the y-axis is pointing to the camera's right, the z-axis is pointing to the camera's downward. You can use the `cam2ned` function in the `evaluation/trajectory_transform.py` to transform the trajectory from the camera frame to the NED frame. 

## Paper
More technical details are available in the [TartanAir paper](https://arxiv.org/abs/2003.14338). Please cite this as: 
```
@article{tartanair2020iros,
  title =   {TartanAir: A Dataset to Push the Limits of Visual SLAM},
  author =  {Wang, Wenshan and Zhu, Delong and Wang, Xiangwei and Hu, Yaoyu and Qiu, Yuheng and Wang, Chen and Hu, Yafei and Kapoor, Ashish and Scherer, Sebastian},
  booktitle = {2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year =    {2020}
}
```

## Contact 
Email tartanair@hotmail.com if you have any questions about the data source. To post problems in the Github issue is also welcome. You can also reach out to contributors on the associated [GitHub](https://github.com/microsoft/AirSim).

## License
[This software is BSD licensed.](https://opensource.org/licenses/BSD-3-Clause)

Copyright (c) 2020, Carnegie Mellon University All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
