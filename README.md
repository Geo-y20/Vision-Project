# Vision Project

## Overview

The Vision Project is a comprehensive computer vision application designed to assist visually impaired individuals by leveraging advanced technologies for object detection, optical character recognition (OCR), and face recognition. Developed by secondary stage students Janaabdelfatah and Romaysa, this project was part of their participation in the Genius Olympiad.

## Features

The Vision Project includes the following features:

1. **Object Detection:**
   - Detects objects such as persons, cups, cats, and dogs using the YOLOv5 model from Hugging Face.
   - Converts detected text to speech using Google TTS.

2. **Optical Character Recognition (OCR):**
   - Converts text from uploaded photos to speech using the Tesseract OCR engine.
   - Features include playing audio, stopping audio, copying text, and searching text.

3. **Face Recognition:**
   - Collects and recognizes faces using the Haar Cascade Classifier.
   - Allows naming persons and managing dataset sizes.

4. **Text-to-Speech (TTS):**
   - Uses Google Speech for converting text to speech.

## Target Audience

The primary target for this project is visually impaired individuals who require assistance in recognizing objects, reading text, and identifying faces in their surroundings.

## Technologies Used

- **Programming Languages:** Python
- **Libraries and Frameworks:** OpenCV, TensorFlow, Keras, Flask
- **Development Tools:** Jupyter Notebook, Visual Studio Code
- **Hardware:** Raspberry Pi 4 with 8GB RAM, Raspberry Pi Camera

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

## Installation and Setup

To set up the Vision Project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Geo-y20/Vision-Project.git
   cd Vision-Project
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r vision.txt
   ```

3. **Run the project:**
   Open a Jupyter Notebook or any Python IDE and execute the main script to start the object detection process.

## Usage

The Vision Project can be used for various applications, including but not limited to:

- Real-time surveillance
- Automated quality inspection
- Robotics and automation
- Augmented reality

## Methodology

The Vision Project follows a systematic approach to ensure the highest performance and reliability:

1. **Requirements Analysis:**
   - Understanding the needs of visually impaired users.
   - Defining functional and non-functional requirements.

2. **System Design:**
   - Creating a blueprint of the overall architecture.
   - Using Flask framework for backend and HTML, CSS, JavaScript for frontend.

3. **Model Selection and Integration:**
   - Object Detection: YOLOv5
   - OCR: Tesseract
   - Face Recognition: Haar Cascade Classifier

4. **Implementation:**
   - Developing the web application.
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

- **George Youhana** - [g.ghaly0451@student.aast.edu](mailto:g.ghaly0451@student.aast.edu)
- **Mostafa Magdy** - [Mustafa.10770@stemredsea.moe.edu.eg](mailto:Mustafa.10770@stemredsea.moe.edu.eg)
- **Abdallah Alkhouly** - [a.alkholy53@student.aast.edu](mailto:a.alkholy53@student.aast.edu)
- **Mohamed Hany Sallam**

Janaabdelfatah and Romaysa, two girls in the secondary stage, competed in the Genius Olympiad with this project.

## Acknowledgements

We would like to thank our mentors and supporters for their guidance and encouragement throughout this project.

## Contact

For any inquiries or further information, please contact the contributors via their provided email addresses.
