# RADIATE SDK

Software development kit (SDK) to use the RADIATE dataset.

## Installation

'pip install -r requirements.txt'

## How to use

The file 'config/config.yaml' controls which sensors to use and configure their parameters.

- **camera_(left\right)_raw**: This is the raw (left\textbackslash right) image captured from the ZED camera with the resolution 672 x 376. For this key, we do not provide the annotation, since the calibration is based on the rectified version. We provide it to the user in case they want to apply their own rectification/calibration method.
- **camera_(left\right)_rect**: This is the rectified (left\textbackslash right) image from the calibration parameters. Since we calibrated the other sensors related to the rectified version, we provide an approximated 2D annotation. We used the distance to the ground and average height of the object to estimate the 2D bounding box. We suppose the measurement is always done in flat roads. We cannot guarantee that the bounding box projection will always occur accurately. Moreover, since the resolution of radar is low (17 cm), the annotation in the camera may not be very precise.
- **radar_polar**: It accesses the radar image in its raw polar format with resolution 400 x 576 (azimuth x range). The index 0 from the azimuth axis represents the angle '0<sup>o<\sup>' and 399 represents the angle '360^o'. Regarding the range axis, index 0 represents 0 meters and index 575 represents 100 meters. This raw format is provided by the sensor manufacturer after applying Fast Fourier Transform (FFT). The manufacturer converts the raw information to decibel (dB), then it is quantised to values between 0 to 255. Therefore, we do not have the raw information in Decibel or Watts. The pixel value represents the power received by the sensor. This value comes mainly from the object material and the shape.     
- **radar_cartesian**: It gives the radar image in cartesian coordinates. We provided a method in the SDK that converts the polar image to a cartesian image by projecting each point onto a (x,y) plane. After projecting each point we use bilinear interpolation to fill the holes without values. This gives an image with *1152 x 1152* image resolution.
- **radar_cartesian_pc**: This item gives the radar cartesian cfar in point cloud format as an 'np.array' with a shape (N,3), where N is the number of points and the columns are (x,y,i), where x and y are the values in meters, and *i* is the intensity power received by the sensor.
- **lidar_pc**: It gives the raw point cloud lidar information in the format (x,y,z,i,r) where x,y,z are the coordinates in meters relative to the radar sensor, 'i' is the power intensity received by the sensor. 'i' is quantised to values between 0 and 255, where it represents mostly the object material. And 'r' says from which ring of the sensor the point came from.
- **lidar_bev_image**: It gives an image with the same size as *radar_cartesian* with a bird's eye view representation. This type of image is created for researchers who want to use the lidar in a grid format and also use it together with the radar in a grid format. 
- **proj_lidar_(left\right)**: This gives the projected lidar points in a camera coordinate frame. It can be used to improve the stereo reconstruction and also fuse the information from the camera with lidar.