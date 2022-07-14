# CNNSpeedTest
Estimate the processing speed of several CNN object detection models.
Using C++ code. Based on the opencv and Nvidia GPU. You can choose GPU or CPU model as you like.

## YOLO series
You can test YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv4-tiny, and so on from the DarkNet weibsite.
Just load the config file and the corresponding weights.

## Faster-RCNN series
We have provided the Faster-RCNN-ResNet50, Faster-RCNN-Inception as the basic model.

| Model name               | GPU NVIDIA GeForce 1080 Ti  | CPU Intel i7-8700k @3.70GHz. |
| ------------------------ | --------------------------- | ---------------------------- |
| YOLOv3-tiny              | 333.58 fps                  | 82.85 fps                    |
| YOLOv3                   | 84.55 fps                   | 12.61 fps                    |
| YOLOv4-tiny              | 257.42 fps                  | 84.02 fps                    |
| YOLOv4                   | 68.36 fps                   | 10.09 fps                    |
| Faster-RCNN-Inception-v2 | 64.28 fps                   | 4.49 fps                     |
| Faster-RCNN-ResNet50     | 33.73 fps                   | 1.37 fps                     |

The image size is 400x340. We run the detector on 1080-ti GPU to statistic the processing time of GPU speed.
Then, we run these model on CPU, Intel Core i7-8700k @3.70GHz to test the speed on CPU.

You can contact me if you have any questions.
(The model and it's config file is easy to find, just goole the name of weights file and config file provided in the c++ code.)
