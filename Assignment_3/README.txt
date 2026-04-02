Small Object Detection Using YOLOv8

This project trains a YOLOv8 model to detect small traffic signs using a subset of the Mapillary Traffic Sign Dataset (MTSD).

Requirements

Install the required Python packages before running the notebook:

pip install ultralytics opencv-python numpy torch torchvision torchaudio

Dataset

The dataset uses YOLO format and is organized under mtsd_yolo_dataset/ with train/, val/, and test/ subdirectories each containing images/ and labels/ folders. The dataset.yaml file defines the dataset path, class count (1 class: traffic-sign), and split locations. Only the validation set is included in this repository due to size. If you want to retrain, you need to download the full Mapillary Traffic Sign Dataset and convert annotations to YOLO format.

How to run

1. Open Small_object.ipynb in Jupyter Notebook or VS Code.
2. Run the cells in order from top to bottom.
3. The notebook first loads the pre-trained yolov8n.pt model and runs initial inference on test images to get baseline predictions.
4. It then fine-tunes the model on the traffic sign dataset with these settings: epochs=20, imgsz=1280, batch=8, patience=10.
5. The best weights are saved during training under the runs/ directory.
6. After training, the notebook runs predictions using the fine-tuned model and compares results against the pre-trained baseline.
7. A second training run with aggressive data augmentation (rotation, scale, shear, translation, flip, mixup, mosaic, color jitter, erasing) is also included for comparison.

Files

Small_object.ipynb - Main notebook with all code for loading, training, and evaluating the model.
yolov8n.pt - Pre-trained YOLOv8 nano weights used as the starting point.
yolo26n.pt - Fine-tuned model weights after training on the traffic sign data.
mtsd_yolo_dataset/dataset.yaml - Dataset configuration file for YOLO training.
pretrained_predictions.json - Initial predictions from the pre-trained model on test images.
runs/ - Training output directory containing logs, metrics, and saved weights.
Homework_3.pdf - Written report with analysis and comparison of results.
LICENSE.txt - Dataset license (CC BY-NC-SA).
