

# Vision Project

## Overview

The Vision Project aims to assist visually impaired individuals by integrating advanced computer vision techniques into a web-based application. This project was developed by secondary stage students Janaabdelfatah and Romaysa, who participated in the Genius Olympiad.

## Features

The Vision Project includes the following features:

1. **Object Detection:**
   - Detects objects such as persons, cups, cats, and dogs using the YOLOv5 model from Hugging Face.

2. **Optical Character Recognition (OCR):**
   - Converts text from uploaded photos to speech using Tesseract OCR.
   - Features include playing audio, stopping audio, copying text, and searching text.

3. **Face Recognition:**
   - Collects and recognizes faces using the Haar Cascade Classifier.
   - Allows naming persons and managing dataset sizes.

4. **Text-to-Speech (TTS):**
   - Uses Google Speech for converting text to speech.

## Target Audience

The primary target for this project is visually impaired individuals who require assistance in recognizing objects, reading text, and identifying faces in their surroundings.

## Installation and Setup

To set up the Vision Project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Geo-y20/Vision-Project.git
   cd Vision-Project
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project:**
   Open a Jupyter Notebook or any Python IDE and execute the main script to start the object detection process.

## Usage

The Vision Project can be used for various applications, including but not limited to:

- Real-time surveillance
- Automated quality inspection
- Robotics and automation
- Augmented reality

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

## OCR: Tesseract

The Tesseract library is used for optical character recognition. For more information, refer to the [Tesseract guide](https://guides.nyu.edu/tesseract/home).

## Face Recognition: Haar Cascade

The Haar Cascade Classifier is used for face detection and recognition. This method involves training a classifier using positive and negative samples and applying it to detect faces in images.

## Contributors

This project was collaboratively developed by the following contributors:

- **George Youhana** - [g.ghaly0451@student.aast.edu](mailto:g.ghaly0451@student.aast.edu)
- **Mostafa Magdy** - [Mustafa.10770@stemredsea.moe.edu.eg](mailto:Mustafa.10770@stemredsea.moe.edu.eg)
- **Abdallah Alkhouly** - [a.alkholy53@student.aast.edu](mailto:a.alkholy53@student.aast.edu)
- **Mohamed Hany Sallam**

Janaabdelfatah and Romaysa, two girls in the secondary stage, competed in the Genius Olympiad with this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

We would like to thank our mentors and supporters for their guidance and encouragement throughout this project.

## Contact

For any inquiries or further information, please contact the contributors via their provided email addresses.
