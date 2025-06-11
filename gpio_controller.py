#!/usr/bin/env python3
import cv2
import os
import time
import numpy as np
from gpiozero import LED
from threading import Thread
from queue import Queue
import sys
os.environ["QT_QPA_PLATFORM"] = "xcb"

# GPIO Pins
RED_PIN = 17      # Pothole (Red LED)
YELLOW_PIN = 27   # Speed Bump (Yellow LED)
GREEN_PIN = 22    # Clear Road (Green LED)

# Constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DETECTION_INTERVAL = 1.0  # seconds
RED_LED_DURATION = 5      # seconds (Pothole)
YELLOW_LED_DURATION = 2   # seconds (Speed Bump)

class GPIOManager:
    """Handles GPIO operations using gpiozero"""
    def __init__(self):
        try:
            self.red_led = LED(RED_PIN)        # Pothole
            self.yellow_led = LED(YELLOW_PIN)  # Speed Bump
            self.green_led = LED(GREEN_PIN)    # Clear Road
            self.gpio_available = True
            print("GPIO initialized:")
            print(f"- Red LED (Pothole) on pin {RED_PIN}")
            print(f"- Yellow LED (Speed Bump) on pin {YELLOW_PIN}")
            print(f"- Green LED (Clear Road) on pin {GREEN_PIN}")
        except Exception as e:
            print(f"GPIO initialization failed: {e}")
            self.gpio_available = False

    def all_off(self):
        if self.gpio_available:
            self.red_led.off()
            self.yellow_led.off()
            self.green_led.off()

    def cleanup(self):
        if self.gpio_available:
            self.all_off()
            print("GPIO cleanup completed.")

    def control_leds(self, detected_labels, led_timer):
        if not self.gpio_available:
            return
        
        current_time = time.time()

        # Update timers based on detections
        if "pothole" in detected_labels:
            led_timer["red"] = current_time + RED_LED_DURATION
        if "Speed Bump" in detected_labels:
            led_timer["yellow"] = current_time + YELLOW_LED_DURATION
        
        # Control LEDs
        red_active = current_time < led_timer["red"]
        yellow_active = current_time < led_timer["yellow"]
        
        self.red_led.on() if red_active else self.red_led.off()
        self.yellow_led.on() if yellow_active else self.yellow_led.off()
        
        # Green LED: ON only if no active detections
        self.green_led.on() if not (red_active or yellow_active) else self.green_led.off()

def load_yolo_model(config_path, weights_path, names_path):
    try:
        print("Loading YOLO model...")
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        with open(names_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        print("YOLO model loaded successfully")
        return net, classes, output_layers
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        raise

def detect_objects(frame, net, output_layers, classes, confidence_threshold=0.5, nms_threshold=0.4):
    height, width = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    class_ids = []
    detected_labels = set()

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                class_ids.append(class_id)
                detected_labels.add(classes[class_id])

    indices = cv2.dnn.NMSBoxes(boxes, [1.0]*len(boxes), confidence_threshold, nms_threshold)
    
    final_boxes = []
    final_ids = []
    
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_ids.append(class_ids[i])
    
    return detected_labels, final_boxes, final_ids

def draw_detections(frame, boxes, class_ids, classes):
    for i, box in enumerate(boxes):
        x, y, w, h = box
        label = str(classes[class_ids[i]])

        if label == "pothole":
            color = (0, 0, 255)  # Red
        elif label == "Speed Bump":
            color = (0, 255, 255)  # Yellow
        else:
            color = (255, 255, 255)  # White

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def capture_frames(cap, frame_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            try:
                frame_queue.put_nowait(frame)
            except:
                pass

def main():
    gpio = GPIOManager()
    
    config_path = "/home/xyz/Desktop/FYP_Data/yolov4-tiny-custom.cfg"
    weights_path = "/home/xyz/Desktop/FYP_Data/yolov4-tiny-custom_final.weights"
    names_path = "/home/xyz/Desktop/FYP_Data/obj.names"

    try:
        net, classes, output_layers = load_yolo_model(config_path, weights_path, names_path)
    except Exception as e:
        gpio.cleanup()
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        gpio.cleanup()
        sys.exit(1)

    print("Road Obstacle Detection System Started. Press 'q' to quit.")

    led_timer = {
        "red": 0,     # Pothole
        "yellow": 0   # Speed Bump
    }

    frame_queue = Queue(maxsize=2)
    capture_thread = Thread(target=capture_frames, args=(cap, frame_queue))
    capture_thread.daemon = True
    capture_thread.start()

    try:
        last_detection_time = 0

        while True:
            try:
                frame = frame_queue.get_nowait()
            except:
                continue

            current_time = time.time()

            if current_time - last_detection_time >= DETECTION_INTERVAL:
                detected_labels, boxes, class_ids = detect_objects(
                    frame, net, output_layers, classes)
                
                print(f"[DEBUG] Detected: {detected_labels}")  # Debug output
                gpio.control_leds(detected_labels, led_timer)
                frame = draw_detections(frame, boxes, class_ids, classes)
                last_detection_time = current_time
            else:
                gpio.control_leds(set(), led_timer)

            cv2.imshow("Road Obstacle Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        gpio.cleanup()
        print("System shutdown complete.")

if __name__ == "__main__":
    main()
