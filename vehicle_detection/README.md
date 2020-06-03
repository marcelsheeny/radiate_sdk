# Radar Vehicle Detection using RADIATE

The file `vehicle_detection.py` has an example radar object detection. It uses RADIATE SDK 

It was trained using [Detectron2](https://github.com/facebookresearch/detectron2) by Facebook AI Research.

Use the [instructions](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) from the original repository to install Detectron2.

## Download weights

Below are the table with the weights trained using radar only. We used Faster R-CNN with rectangular and rotated bounding boxes. We also provided training in both good and bad weather and in good weather only.
The results (AP) are computed from the training set which contains both good and bad weather data.

### Rectangular Bounding Boxes
| Network        | Download Command           | AP(%)  |
| ------------- |-------------| -----|
| `faster_rcnn_resnet50` Trained in good and bad weather     | `wget -O weights/faster_rcnn_R_50_FPN_3x_good_and_bad_weather_radar.pth https://www.dropbox.com/s/ltwf6qiv49sn20m/model_final.pth?dl=0` | 53.57 |
| `faster_rcnn_resnet50` Trained in good weather only   | `wget -O weights/faster_rcnn_R_50_FPN_3x_good_weather_radar.pth https://www.dropbox.com/s/ou05i3os1idlll9/model_final.pth?dl=0`      |   52.77 |
| `faster_rcnn_resnet101` Trained in good and bad weather| `wget -O weights/faster_rcnn_R_101_FPN_3x_good_and_bad_weather_radar.pth https://www.dropbox.com/s/qst7he00mpkvi44/model_final.pth?dl=0`     |   54.43 |
|`faster_rcnn_resnet101` Trained in good weather only |`wget -O weights/faster_rcnn_R_101_FPN_3x_good_weather_radar.pth https://www.dropbox.com/s/8ab4ixp5s9bhns2/model_final.pth?dl=0`| 52.90|




### Rotated Bounding Boxes
| Network        | Download Command           | AP(%)  |
| ------------- |-------------| -----|
| `faster_rcnn_resnet50` Trained in good and bad weather      | `wget -O weights/faster_rrcnn_R_50_FPN_3x_good_and_bad_weather_radar.pth https://www.dropbox.com/s/eoj83ynvs1dernx/model_final.pth?dl=0` | 45.77 |
| `faster_rcnn_resnet50` Trained in good weather only      | `wget -O weights/faster_rrcnn_R_50_FPN_3x_good_weather_radar.pth https://www.dropbox.com/s/iwpisqpa9y2nfng/model_final.pth?dl=0`      |   45.31 |
| `faster_rcnn_resnet101` Trained in good and bad weather| `wget -O weights/faster_rrcnn_R_101_FPN_3x_good_and_bad_weather_radar.pth https://www.dropbox.com/s/woxp48qsooupqpl/model_final.pth?dl=0`     |   46.55 |
|`faster_rcnn_resnet101` Trained in good weather only|`wget -O weights/faster_rrcnn_R_101_FPN_3x_good_weather_radar.pth https://www.dropbox.com/s/hg7z2jznipf320c/model_final.pth?dl=0`| 45.84|














