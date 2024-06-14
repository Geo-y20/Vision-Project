
# Vision Project

## Overview

The Vision Project is a comprehensive computer vision application designed to assist visually impaired individuals by leveraging advanced technologies for object detection, optical character recognition (OCR), and face recognition. Idea by secondary stage students Janaabdelfatah and Romaysa, this project was part of their participation in the Genius Olympiad.

## Video Demonstration

Check out this video demonstration of the Vision Project in action:
    
[![Vision Project Video](https://github.com/Geo-y20/Vision-Project/blob/main/thumbnail.JPG)](https://github.com/Geo-y20/Vision-Project/blob/main/Final-obj-ocr-face-project.mp4)

## Features

### Version 1: Website (Using Laptop)

- **Technologies Used:**
  - Flask framework for backend.
  - HTML, CSS, and JavaScript for frontend.
  - Python for core functionality.

- **Functionalities:**
  - **Object Detection:** Uses YOLOv5 model from Hugging Face.
  - **Face Recognition:** Uses Haar Cascade Classifier.
  - **Optical Character Recognition (OCR):** Uses Tesseract OCR.
  - **Text-to-Speech (TTS):** Converts detected text to speech using Google TTS.

### Version 2: Raspberry Pi

- **Technologies Used:**
  - Python for core functionality.
  - Raspberry Pi 4 with 8GB RAM and Raspberry Pi Camera.

- **Functionalities:**
  - **Object Detection:** Uses YOLOv5 model.
  - **Face Detection:** Due to computational power limitations, uses Haar Cascade Classifier.
  - **Optical Character Recognition (OCR):** Uses Tesseract OCR.

## Hardware Details

### Raspberry Pi 4 Model B (8 GB)

- **Description:**
  - Raspberry Pi 4 Model B, Wi-Fi, 2x micro HDMI, USB-C, USB 3.0, 8 GB of RAM 1.5 GHz.
  - The latest product in the Raspberry Pi range, offering improvements in processor speed, multimedia performance, memory, and connectivity.

- **Main Features:**
  - 64-bit quad-core processor.
  - Dual display support with resolutions up to 4K.
  - 8GB LPDDR4-2400 SDRAM.
  - Dual-band 2.4/5.0 GHz wireless LAN, Bluetooth 5.0, Gigabit Ethernet.
  - USB 3.0 and PoE capabilities (via a separate PoE HAT add-on).

### Raspberry Pi Camera Board v1.3

- **Description:**
  - Plugs directly into the CSI connector on the Raspberry Pi.
  - Delivers a 5MP resolution image or 1080p HD video recording at 30fps.

## Installation and Setup

### Version 1: Website (Using Laptop)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Geo-y20/Vision-Project.git
   cd Vision-Project
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r vision.txt
   ```

3. **Run the Flask application:**
   ```bash
   flask run
   ```

### Version 2: Raspberry Pi

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Geo-y20/Vision-Project.git
   cd Vision-Project
   ```

2. **Ensure the Raspberry Pi environment is correctly set up with all necessary packages installed.**

3. **Run the scripts:**

   - **camera.py:** Check for camera functionality.
     ```bash
     python camera.py
     ```

   - **facedetection.py:** Perform face detection using the Haar Cascade Classifier.
     ```bash
     python facedetection.py
     ```

   - **obj.py:** Perform object detection using YOLOv5.
     ```bash
     python obj.py
     ```

   - **ocr.py:** Perform OCR using Tesseract.
     ```bash
     python ocr.py
     ```

## Scripts Explanation

- **camera.py:**
  - Checks if the Raspberry Pi camera is correctly set up and functional.
  - Ensures the camera can capture images and video.

- **facedetection.py:**
  - Uses the Haar Cascade Classifier to detect faces in real-time.
  - Captures video from the camera and applies the face detection algorithm.

- **obj.py:**
  - Uses YOLOv5 for real-time object detection.
  - Captures video from the camera, processes it through the YOLOv5 model, and identifies objects.

- **ocr.py:**
  - Uses Tesseract to perform OCR on images captured by the camera.
  - Converts the recognized text to speech using Google TTS.

## Object Detection: YOLOv5

YOLOv5 (You Only Look Once) is used for real-time object detection. For more details on YOLOv5, visit the [Roboflow blog](https://blog.roboflow.com/yolov5-improvements-and-evaluation/) and the [COCO dataset](https://cocodataset.org/#home).

### Precision and Recall Equations
- **Precision:** \( \text{Precision} = \frac{TP}{TP + FP} \)
  - TP: True Positives
  - FP: False Positives
- **Recall:** \( \text{Recall} = \frac{TP}{TP + FN} \)
  - TP: True Positives
  - FN: False Negatives

### Object Detection Performance with YOLOv5

| Object     | Precision (%) | Recall (%) | Processing Time (ms) |
|------------|----------------|------------|----------------------|
| Person     | 98             | 97         | 20                   |
| Car        | 96             | 95         | 22                   |
| Bicycle    | 95             | 93         | 25                   |
| Dog        | 94             | 92         | 23                   |
| Cat        | 93             | 91         | 24                   |

## OCR: Tesseract

The Tesseract library is used for optical character recognition. For more information, refer to the [Tesseract guide](https://guides.nyu.edu/tesseract/home).

### OCR Performance

| Document Type | Precision (%) | Recall (%) | Processing Time (ms) |
|---------------|----------------|------------|----------------------|
| Invoice       | 95             | 94         | 150                  |
| Letter        | 93             | 92         | 140                  |
| Receipt       | 94             | 91         | 145                  |
| Book Page     | 92             | 90         | 155                  |
| ID Card       | 90             | 88         | 160                  |

## Face Recognition: Haar Cascade

The Haar Cascade Classifier is used for face detection and recognition. This method involves training a classifier using positive and negative samples and applying it to detect faces in images.

### Face Recognition Performance

| Person   | Precision (%) | Recall (%) | Processing Time (ms) |
|----------|----------------|------------|----------------------|
| Jana     | 98             | 97         | 100                  |
| Romaysa  | 97             | 96         | 105                  |
| Mariam   | 96             | 95         | 110                  |
| Mohamed  | 95             | 94         | 115                  |
| Youssef  | 94             | 93         | 120                  |

## Methodology

The Vision Project follows a systematic approach to ensure the highest performance and reliability:

1. **Requirements Analysis:**
   - Understanding the needs of visually impaired users.
   - Defining functional and non-functional requirements.

2. **System Design:**
   - Creating a blueprint of the overall architecture.
   - Using Flask framework for backend and HTML, CSS, JavaScript for frontend in the laptop version.
   - Using Python for core functionality in the Raspberry Pi version.

3. **Model Selection and Integration:**
   - Object Detection: YOLOv5
   - OCR: Tesseract
   - Face Recognition: Haar Cascade Classifier

4. **Implementation:**
   - Developing the web application for the laptop version.
   - Integrating the models for object detection, OCR, and face recognition.

5. **Testing:**
   - Unit Testing: Testing individual components.
   - Integration Testing: Ensuring all components work together.
   - Performance Testing: Measuring response times and accuracy.
   - User Testing: Gathering feedback from visually impaired users.

6. **Evaluation:**
   - Analyzing performance metrics.
   - Visualizing results using graphs and charts.

### Additional Graphs and Charts

- **Confusion Matrix:**
  For each task (Object Detection, OCR, Face Recognition), a confusion matrix shows the performance in terms of true positives, false positives, false negatives, and true negatives.

- **Precision-Recall Curve:**
  Shows the trade-off between precision and recall for different threshold settings.

- **Receiver Operating Characteristic (ROC) Curve:**
  Plots the true positive rate against the false positive rate for binary classification tasks.

- **F1 Score:**
  Combines precision and recall into a single metric using the harmonic mean.

- **Accuracy Over Different Conditions:**
  Compares accuracy under various conditions such as different lighting or image quality levels.

## Contributors

This project was collaboratively developed by the following contributors:

- **George Youhana** - [georgeyouhana2@gmail.com](mailto:georgeyouhana2@gmail.com)
- **Mostafa Magdy** - [Mustafa.10770@stemredsea.moe.edu.eg](mailto:Mustafa.10770@stemredsea.moe.edu.eg)
- **Abdallah Alkhouly** - [a.alkholy53@student.aast.edu](mailto:a.alkholy53@student.aast.edu)
- **Mohamed Hany Sallam** - [m.h.sallam1@student.aast.edu](mailto:m.h.sallam1@student.aast.edu)

Janaabdelfatah and Romaysa, two girls in the secondary stage, competed in the Genius Olympiad with this project.

## Access the Project

You can access the project files here: [raspberry pi.rar](https://github.com/Geo-y20/Vision-Project/blob/main/raspberry%20pi.rar)

## Contact

For any inquiries or further information, please contact the contributors via their provided email addresses.
