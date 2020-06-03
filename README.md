# RADIATE Dataset

RADIATE (RAdar Dataset In Adverse weaThEr) is new automotive dataset created by Heriot-Watt University which includes Radar, Lidar, Stereo Camera and GPS/IMU.

We collected data in different weather scenarios (sunny, overcast, night, fog, rain and snow) to help the research community to develop new methods of vehicle perception

![](assets/radiate.png)

## Sensors

![](assets/sensors.png)

* **Stereo Camera**: An off-the-shelf ZED stereo camera is used. It is set at 672 × 376 image resolution at 15 frames per second for each camera. It is protected by a waterproof housing
for extreme weather. The images can be seriously blurred, hazy or fully blocked due to rain drops, dense fog or heavy snow, respectively.
* **LiDAR**: A 32 channel, 10Hz, Velodyne HDL-32e LiDAR is used to give 360° coverage. Since the LiDAR signal can be severely attenuated and reflected by intervening fog or snow the data can be missing, noisy and incorrect.
* **Radar** RADIATE adopts the Navtech CTS350-X radar. It is a scanning radar which provides 360° high-resolution range-azimuth images. It has 100 meters maximum operating range with 0.175m range resolution, 1.8° azimuth resolution and 1.8° elevation resolution, Currently, it does not provide Doppler information.
* **GPS/IMU** : Advanced Navigation GPS/IMU is provided. 

## Folder Structure and File Format

![](assets/folder_tree.png)

* **GPS_IMU_Twist**: We provide a readable *.txt* file. Each line is shown below:
  
    GPS-IMU file format:

    Latitude Longitude Altitude

    GPS position_covariance (3x3)

    IMU.Orientation.X IMU.Orientation.Y IMU.Orientation.Z IMU.Orientation.W

    IMU.AngularVelocity.X IMU.AngularVelocity.Y IMU.AngularVelocity.Z

    IMU.LinearAccelaration.X IMU.LinearAccelaration.Y IMU.LinearAccelaration.Z

    IMU.Orientation Covariance (3x3)

    IMU.AngularVelocity Covariance (3x3)

    IMU.LinearAccelaration Covariance (3x3)

* **Navtech_Polar**: We provide *.png* with resolution 400 x 576. Where each row represents the range with resolution 0.17361 m. And each column represents the angle with resolution 1.1°.

* **Navtech_Cartesian**: We provide *.png* with resolution 1152 x 1152. This is a implementation from polar to cartesian. We use nearest neighbor interpolation. The each pixel represents a 0.17361 m x 0.17361 m.
*  **velo_lidar**: We provide readable *.txt* files where each line represents x,y,z,intensity,ring. (x,y,z) represents the 3D point cloud in the lidar frame. Intensity [0-255] is reflectance captured by the sensor. Ring [1-32] means from each of the 32 channels the detected point came from.
*  **zed_left/right**: We provide *.png* unrectified images with resolution 672 × 376. 
  
Each folder contains a *FOLDER.txt* which shows the timestamp for each collected frame. The format is Frame: XXXXXX Time: XXXXXX. Where frame is the frame ID which correspond to the filename. Time is the timestamp using UNIX time system in seconds.


## Annotation Structure

The annotation is a *.json* file. where each entry contains *id,class_name,bboxes*. *id* is the object identification. *class_name* is a string with the name class. *bboxes* contains *position*: (x,y,width,height) where (x,y) is the upper-left pixel locations of the bounding box, of given width and height. And *angle* is the angle in degrees using counter-clockwise.

# RADIATE SDK

Software development kit (SDK) to use the RADIATE dataset. The SDK was tested using Python 3.7. The SDK is used for data calibration, visualisation, and pre-processing.

## Installation

```
git clone https://github.com/marcelsheeny/radiate_sdk.git
cd radiate_sdk
pip install -r requirements.txt
```

Run `demo.py` to visualise the dataset.

### Dependencies
```
matplotlib
opencv-python
pandas
numpy
pyyaml
```

## How to use

The file 'config/config.yaml' controls which sensors to use and configure their parameters.

- **camera_(left\right)_raw**: This is the raw (left\textbackslash right) image captured from the ZED camera with the resolution 672 x 376. For this key, we do not provide the annotation, since the calibration is based on the rectified version. We provide it to the user in case they want to apply their own rectification/calibration method.
- **camera_(left\right)_rect**: This is the rectified (left\textbackslash right) image from the calibration parameters. Since we calibrated the other sensors related to the rectified version, we provide an approximated 2D annotation. We used the distance to the ground and average height of the object to estimate the 2D bounding box. We suppose the measurement is always done in flat roads. We cannot guarantee that the bounding box projection will always occur accurately. Moreover, since the resolution of radar is low (17 cm), the annotation in the camera may not be very precise.
- **radar_polar**: It accesses the radar image in its raw polar format with resolution 400 x 576 (azimuth x range). The index 0 from the azimuth axis represents the angle '0<sup>o</sup>' and 399 represents the angle '360<sup>o</sup>'. Regarding the range axis, index 0 represents 0 meters and index 575 represents 100 meters. This raw format is provided by the sensor manufacturer after applying Fast Fourier Transform (FFT). The manufacturer converts the raw information to decibel (dB), then it is quantised to values between 0 to 255. Therefore, we do not have the raw information in Decibel or Watts. The pixel value represents the power received by the sensor. This value comes mainly from the object material and the shape.     
- **radar_cartesian**: It gives the radar image in cartesian coordinates. We provided a method in the SDK that converts the polar image to a cartesian image by projecting each point onto a (x,y) plane. After projecting each point we use bilinear interpolation to fill the holes without values. This gives an image with *1152 x 1152* image resolution.
- **radar_cartesian_pc**: This item gives the radar cartesian cfar in point cloud format as an 'np.array' with a shape (N,3), where N is the number of points and the columns are (x,y,i), where x and y are the values in meters, and *i* is the intensity power received by the sensor.
- **lidar_pc**: It gives the raw point cloud lidar information in the format (x,y,z,i,r) where x,y,z are the coordinates in meters relative to the radar sensor, 'i' is the power intensity received by the sensor. 'i' is quantised to values between 0 and 255, where it represents mostly the object material. And 'r' says from which ring of the sensor the point came from.
- **lidar_bev_image**: It gives an image with the same size as *radar_cartesian* with a bird's eye view representation. This type of image is created for researchers who want to use the lidar in a grid format and also use it together with the radar in a grid format. 
- **proj_lidar_(left\right)**: This gives the projected lidar points in a camera coordinate frame. It can be used to improve the stereo reconstruction and also fuse the information from the camera with lidar.

The file `demo.py` contains a small code which just display the annotations.

```
import radiate
import numpy as np
import os

# path to the sequence
root_path = 'data/radiate/'
sequence_name = 'fog_6_0'

# time (s) to retrieve next frame
dt = 0.25

# load sequence
seq = radiate.Sequence(os.path.join(root_path, sequence_name))

# play sequence
for t in np.arange(seq.init_timestamp, seq.end_timestamp, dt):
    output = seq.get_from_timestamp(t)
    seq.vis_all(output, 0)
```

In order to get the annotation values, the variable 'output' is a dictionary with the sensor and its correspondent annotation.

### Example: 

`output['sensors']['radar_cartesian']` contains a np.array with the radar image.

`output['annotations']['radar_cartesian']` contains a list of bounding boxes with `id`, `class_name` and `bbox`. `bbox` : `position` is represented as `x,y,width,height` and `bbox` : `rotation` is the angle counter-clockwise in degrees. This is exemplified below:

```
'id':1
'class_name':'bus'
'bbox':{'position': [603.5340471042896, 149.7590074419735, 26.620884098218767, 73.56976270380676], 'rotation': 177.69489304897752}
__len__:3
```

The documentation of all radiate methods can be seen at:
https://marcelsheeny.github.io/radiate_sdk/radiate.html