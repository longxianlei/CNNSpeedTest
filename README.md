# CNNSpeedTest
Estimate the processing speed of several CNN object detection models.
Using C++ code. Based on the opencv and Nvidia GPU. You can choose GPU or CPU model as you like.

## YOLO series
You can test YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv4-tiny, and so on from the DarkNet weibsite.
Just load the config file and the corresponding weights.

## Faster-RCNN series
We have provided the Faster-RCNN-ResNet50, Faster-RCNN-Inception as the basic model.

| Model name               | GPU  NVIDIA GeForce 1080 Ti | CPU Intel i7-8700k @3.70GHz. |
| ------------------------ | --------------------------- | ---------------------------- |
| YOLOv3-tiny              | 333.58 fps                  | 82.85 fps                    |
| YOLOv3                   | 84.55 fps                   | 12.61 fps                    |
| YOLOv4-tiny              | 257.42 fps                  | 84.02 fps                    |
| YOLOv4                   | 68.36 fps                   | 10.09 fps                    |
| Faster-RCNN-Inception-v2 | 64.28 fps                   | 4.49 fps                     |
| Faster-RCNN-ResNet50     | 33.73 fps                   | 1.37 fps                     |

The image size is 400x340
