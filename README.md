📷 Live Camera Dictionary (Python + YOLOv8)

A real-time computer vision application that uses a webcam and the YOLOv8 object detection model to identify everyday objects instantly. The system highlights the object closest to the center of the camera and displays its name on the screen.

✨ Features
Real-time object detection using YOLOv8
Live webcam video processing
Detects and labels the object nearest to the camera center
Confidence-based object filtering
Clean and responsive interface
Supports detection of 80+ common object classes (COCO dataset)
🛠 Tech Stack

Python, OpenCV, Ultralytics YOLOv8

⚙️ How It Works

The application captures live video from the webcam and processes each frame with a pretrained YOLOv8 model. Among all detected objects, it selects the one closest to the center of the frame and displays its bounding box and label in real time.
