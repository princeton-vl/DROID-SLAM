### GRB Image

The color images are stored as 640x480 8-bit RGB images in PNG format.

* Load the image using OpenCV: 
```
import cv2
img = cv2.imread(FILENAME)
cv2.imshow('img', img)
cv2.waitKey(0)
```

* Load the image using Pillow:
```
from PIL import Image
img = Image.open(FILENAME)
img.show()
```

### Camera intrinsics 
```
fx = 320.0  # focal length x
fy = 320.0  # focal length y
cx = 320.0  # optical center x
cy = 240.0  # optical center y

fov = 90 deg # field of view

width = 640
height = 480
```

### Depth image

The depth maps are stored as 640x480 16-bit numpy array in NPY format. In the Unreal Engine, the environment usually has a sky sphere at a large distance. So the infinite distant object such as the sky has a large depth value (e.g. 10000) instead of an infinite number. 

The unit of the depth value is meter. The baseline between the left and right cameras is 0.25m. 

* Load the depth image:
```
import numpy as np
depth = np.load(FILENAME)

# change to disparity image
disparity = 80.0 / depth
```

### Segmentation image

The segmentation images are saved as a uint8 numpy array. AirSim assigns value 0 to 255 to each mesh available in the environment. 

[More details](https://github.com/microsoft/AirSim/blob/master/docs/image_apis.md#segmentation)

* Load the segmentation image
```
import numpy as np
depth = np.load(FILENAME)
```

### Optical flow

The optical flow maps are saved as a float32 numpy array, which is calculated based on the ground truth depth and ground truth camera motion, using [this](https://github.com/huyaoyu/ImageFlow) code. Dynamic objects and occlusions are masked by the mask file, which is a uint8 numpy array. We currently provide the optical flow for the left camera. 

* Load the optical flow
```
import numpy as np
flow = np.load(FILENAME)

# load the mask
mask = np.load(MASKFILENAME)
```

### Pose file

The camera pose file is a text file containing the translation and orientation of the camera in a fixed coordinate frame. Note that our automatic evaluation tool expects both the ground truth trajectory and the estimated trajectory to be in this format. 

* Each line in the text file contains a single pose.

* The number of lines/poses is the same as the number of image frames in that trajectory. 

* The format of each line is '**tx ty tz qx qy qz qw**'. 

* **tx ty tz** (3 floats) give the position of the optical center of the color camera with respect to the world origin in the world frame.

* **qx qy qz qw** (4 floats) give the orientation of the optical center of the color camera in the form of a unit quaternion with respect to the world frame. 

* The camera motion is defined in the NED frame. That is to say, the x-axis is pointing to the camera's forward, the y-axis is pointing to the camera's right, the z-axis is pointing to the camera's downward. 

* Load the pose file:
```
import numpy as np
flow = np.loadtxt(FILENAME)
```
