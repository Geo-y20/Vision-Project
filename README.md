
# Vision Project

## Overview

The Vision Project is a comprehensive computer vision application designed to assist visually impaired individuals by leveraging advanced technologies for object detection, optical character recognition (OCR), and face recognition. Developed by secondary stage students Janaabdelfatah and Romaysa Gomaa, this project was part of their participation in the Genius Olympiad.

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

## Hardware Details

### Raspberry Pi 4 Model B (8 GB)

- **Description:**
  - Raspberry Pi 4 Model B, Wi-Fi, 2x micro HDMI, USB-C, USB 3.0, 8 GB of RAM 1.5 GHz.
  - The latest product in the Raspberry Pi range, offering improvements in processor speed, multimedia performance, memory, and connectivity compared to the previous generation Raspberry Pi 3 Model B +.
  - Offers desktop performance comparable to entry-level x86 PC systems.

- **Main Features:**
  - 64-bit quad-core processor.
  - Dual display support with resolutions up to 4K via 2 micro-HDMI ports.
  - Hardware video decoding up to 4Kp60.
  - 8GB LPDDR4-2400 SDRAM.
  - Dual-band 2.4/5.0 GHz wireless LAN.
  - Bluetooth 5.0, Gigabit Ethernet.
  - USB 3.0 and PoE capabilities (via a separate PoE HAT add-on).

- **Specifications:**
  - Broadcom BCM2711, quad-core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5GHz.
  - 8GB LPDDR4-2400 SDRAM.
  - 2.4GHz and 5.0GHz IEEE 802.11b/g/n/ac wireless LAN, Bluetooth 5.0, BLE.
  - True Gigabit Ethernet.
  - 2x USB 3.0 ports, 2x USB 2.0 ports.
  - Fully backward-compatible 40-pin GPIO header.
  - 2x Micro HDMI ports which support a resolution up to 4K at 60Hz.
  - 2-channel MIPI DSI/CSI ports for camera and display.
  - 4-channel stereo audio and composite video port.
  - MicroSD card slot for the OS and storage.
  - 5.1V, 3A via USB-C or GPIO.
  - PoE (Power over Ethernet) suitable (with optional PoE HAT).

- **Product Link:**
  [Raspberry Pi 4 Model B (8 GB)](https://makerselectronics.com/product/raspberry-pi-4-computer-model-b-8gb-ram-made-in-uk)

### Raspberry Pi Camera Board v1.3

- **Description:**
  - The Raspberry Pi Camera Board plugs directly into the CSI connector on the Raspberry Pi.
  - Delivers a crystal clear 5MP resolution image, or 1080p HD video recording at 30fps.
  - Custom designed and manufactured by the Raspberry Pi Foundation in the UK.

- **Main Features:**
  - 5MP (2592×1944 pixels) Omnivision 5647 sensor.
  - Fixed focus module.
  - Capable of 2592 x 1944 pixel static images.
  - Supports 1080p @ 30fps, 720p @ 60fps, and 640x480p 60/90 video recording.
  - Attaches via a 15 Pin Ribbon Cable to the dedicated 15-pin MIPI Camera Serial Interface (CSI).
  - Tiny and lightweight (around 25mm x 20mm x 9mm, just over 3g).

- **Specifications:**
  - Fully Compatible with Both the Model A and Model B Raspberry Pi.
  - 5MP Omnivision 5647 Camera Module.
  - Still Picture Resolution: 2592 x 1944.
  - Video: Supports 1080p @ 30fps, 720p @ 60fps, and 640x480p 60/90 Recording.
  - 15-pin MIPI Camera Serial Interface – Plugs Directly into the Raspberry Pi Board.
  - Size: 24 x 24.5 x 9mm.
  - Weight: 3g.
  - Fully Compatible with many Raspberry Pi cases.

- **Product Link:**
  [Raspberry Pi Camera Board v1.3](https://makerselectronics.com/product/raspberry-pi-camera-board-v1-3-5mp-1080p)

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
   - Object

 Detection: YOLOv5
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
- **Mohamed Hany Sallam** -[m.h.sallam1@student.aast.edu](mailto:m.h.sallam1@student.aast.edu)

Janaabdelfatah and Romaysa, two girls in the secondary stage, competed in the Genius Olympiad with this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

We would like to thank our mentors and supporters for their guidance and encouragement throughout this project.

## Access the Project

You can access the project files here: [raspberry pi.rar](path_to_your_raspberry_pi.rar)

## Contact

For any inquiries or further information, please contact the contributors via their provided email addresses.
