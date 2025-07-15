-----

# 🚀 Real-Time Driver Drowsiness Detection using YOLOv8

This project implements a real-time driver drowsiness detection system using a custom-trained **YOLOv8** object detection model. The system captures video from a webcam, processes each frame to detect the driver's state (drowsy or awake), and displays the results with bounding boxes and descriptive labels.

## ✨ Features

  - **Real-Time Detection**: Analyzes live webcam feed to monitor the driver.
  - **High Accuracy**: Utilizes a custom-trained YOLOv8 model for robust detection.
  - **Visual & Textual Feedback**: Draws bounding boxes around the detected person and labels their state as 'Drowsy' or 'Awake'.
  - **Performance Metrics**: Displays the real-time Frames Per Second (FPS) of the detection loop.

-----

## 🛠️ Technologies Used

This project is built with the following technologies:

  - **Python**
  - **PyTorch**
  - **Ultralytics (YOLOv8)**
  - **OpenCV**
  - **NumPy**

-----

## 📋 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Make sure you have Python 3.8+ installed on your system. You will also need to install the required Python libraries.

### Installation

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/your-username/drowsiness-detection.git
    cd drowsiness-detection
    ```

2.  **Create a virtual environment (Recommended):**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    Create a file named `requirements.txt` with the following content:

    ```txt
    ultralytics
    opencv-python
    numpy
    ```

    Then, install the packages using pip:

    ```sh
    pip install -r requirements.txt
    ```

4.  **Download Model Weights:**
    You must have the trained model weights file named **`best.pt`**. Place this file in the root directory of the project.

-----

## ▶️ Usage

The core logic from the Jupyter Notebook has been adapted into a runnable Python script.

1.  Create a file named `detect.py` and paste the code below into it.
2.  Ensure your webcam is connected. The script might need adjustment for the correct webcam index (`capture_index=0` is usually the default built-in webcam).
3.  Run the script from your terminal:
    ```sh
    python detect.py
    ```

The script will open a window displaying your webcam feed with the drowsiness detection results overlaid. Press `q` to quit the application.

### `detect.py`

```python
import cv2
import numpy as np
from ultralytics import YOLO
from time import time

class ObjectDetection:
    """
    A class for performing real-time object detection on a video stream using YOLOv8.
    """
    def __init__(self, capture_index):
        """
        Initializes the ObjectDetection object.

        Args:
            capture_index (int): The index of the video capture device.
        """
        self.capture_index = capture_index
        self.model = self.load_model()

    def load_model(self):
        """
        Loads the YOLOv8 model from the 'best.pt' file.

        Returns:
            A YOLO model object.
        """
        model = YOLO("best.pt")
        return model

    def predict(self, frame):
        """
        Performs object detection on a single frame.

        Args:
            frame (np.ndarray): The input frame from the video stream.

        Returns:
            The results of the object detection.
        """
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):
        """
        Plots bounding boxes and labels on the frame.

        Args:
            results: The detection results from the YOLO model.
            frame (np.ndarray): The frame to draw on.

        Returns:
            np.ndarray: The frame with bounding boxes and labels.
        """
        xyxys = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        class_names = results[0].names

        for xyxy, conf, cls_id in zip(xyxys, confidences, class_ids):
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"{class_names[cls_id]}: {conf:.2f}"
            
            # Choose color based on class
            if class_names[cls_id].lower() == 'drowsy':
                color = (0, 0, 255) # Red for drowsy
            else:
                color = (0, 255, 0) # Green for awake/other

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame

    def __call__(self):
        """
        The main loop for capturing frames, performing detection, and displaying results.
        """
        cap = cv2.VideoCapture(self.capture_index)
        if not cap.isOpened():
            print(f"Error: Could not open video capture device at index {self.capture_index}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            start_time = time()

            ret, frame = cap.read()
            if not ret:
                break

            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)

            end_time = time()
            fps = 1 / (end_time - start_time)

            cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('YOLOv8 Drowsiness Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use 0 for default webcam, 1 for external, etc.
    detector = ObjectDetection(capture_index=0)
    detector()
```

-----

## 🧠 How It Works

This system operates by feeding frames from a webcam into a custom-trained YOLOv8 model. Unlike traditional methods that rely on calculating specific metrics like the Eye Aspect Ratio (EAR), this deep learning model directly analyzes the entire frame to make a holistic prediction.

1.  **Frame Capture**: The webcam captures a video frame.
2.  **Model Inference**: The frame is passed to the `best.pt` YOLOv8 model.
3.  **Output**: The model outputs bounding boxes, confidence scores, and class predictions.
4.  **Visualization**: The results are drawn onto the original frame and displayed, providing immediate visual feedback.

This end-to-end approach allows the model to learn complex features indicative of drowsiness—such as head posture, yawning, and eye state—without explicit programming.

-----

## 🤖 NLP and Computer Vision Synergy

While this project is fundamentally a **Computer Vision** task, it intersects with **Natural Language Processing (NLP)** in a crucial way: **semantic labeling**.

The core function of the model is to map complex visual data (pixels representing a person's face and posture) to a human-readable, linguistic concept.

  - The model doesn't just output a category index like `0` or `1`.
  - It associates these indices with meaningful text labels: **'Drowsy'** and **'Awake'**.

This process of assigning descriptive text to visual phenomena is a bridge between CV and NLP. The model learns to translate visual patterns into natural language, making its output instantly interpretable and actionable for a human observer. In essence, the NLP component gives meaning and context to the visual detection.

-----

## 📄 License

This project is distributed under the MIT License. See the `LICENSE` file for more information.
