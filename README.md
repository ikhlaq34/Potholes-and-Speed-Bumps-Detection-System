🚗 Pothole and Speed Bump Detection System

This project is a real-time **road obstacle detection system** designed to identify **potholes**, **speed bumps**, and **clear roads** using a camera and a YOLO-based object detection model. It aims to improve driving safety and comfort by providing visual alerts using **LED indicators**.

---

 📌 Features

* Real-time object detection using a mounted camera
* Detection of:

  * Potholes
  * Speed bumps
  * Clear roads
* LED alert system via GPIO pins
* Lightweight implementation for embedded systems

---

🛠️ Technologies Used

Python
OpenCV
YOLOv5 (You Only Look Once)
PyTorch
Raspberry Pi (or any embedded system with GPIO support)
GPIO for LED interfacing
Tkinter (optional GUI interface)

---

 ⚙️ Hardware Requirements

* Raspberry Pi / Jetson Nano / compatible microcontroller
* Pi-compatible camera (e.g., Pi Camera v2 or USB webcam)
* Breadboard and jumper wires
* 3 LEDs (Red for pothole, Yellow for speed bump, Green for clear road)
* Resistors (220 ohm)

---

 📁 Project Structure

```
pothole-detection/
│
├── yolov5/                  # YOLOv5 model and scripts
├── dataset/                 # Custom annotated dataset (YOLO format)
├── weights/                 # Trained model weights (.pt file)
├── main.py                  # Main application script
├── gpio_controller.py       # Controls GPIO LEDs based on detection
├── detect.py                # Custom YOLO detect function
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pothole-detection.git
cd pothole-detection
```

 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure you also have PyTorch and OpenCV installed. You may need to configure them based on your hardware (e.g., CUDA support).

 3. Run the Detection Script

```bash
python main.py
```

This will start the camera, detect road conditions in real-time, and light up the corresponding LED:

* 🔴 Red = Pothole
* 🟡 Yellow = Speed bump
* 🟢 Green = Clear road

---

 🧠 Model Training (Optional)

To retrain the model on your own dataset:

```bash
cd yolov5
python train.py --img 640 --batch 16 --epochs 50 --data pothole.yaml --weights yolov5s.pt
```

---

 📊 Performance

| Obstacle   | Precision | Recall | mAP\@0.5 |
| ---------- | --------- | ------ | -------- |
| Pothole    | 92.3%     | 88.7%  | 90.2%    |
| Speed Bump | 90.5%     | 85.6%  | 88.1%    |
| Clear Road | 95.1%     | 93.8%  | 94.2%    |

> *(Note: Values are based on a custom-trained YOLOv5s model)*

---

 📸 Demo

You can add a video or GIF showing the working system in real-time here.

---

 👨‍💻 Authors

Ikhlaq Ahmed – [LinkedIn](https://www.linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

---

 📜 License

This project is licensed under the MIT License 

