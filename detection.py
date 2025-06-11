import cv2 as cv
import time
import geocoder
import os

# Define absolute paths
base_path = r"C:\Users\akhal\Downloads\pothole-detection-main\pothole-detection-main\project_files"
# Define absolute paths directly
obj_names_path = r"C:\Users\akhal\OneDrive\Documents\FYP\pothole-detection-main\pothole-detection-main\project_files\obj.names"
weights_path = r"C:\Users\akhal\OneDrive\Documents\FYP\pothole-detection-main\pothole-detection-main\project_files\yolov4_tiny.weights"
config_path = r"C:\Users\akhal\OneDrive\Documents\FYP\pothole-detection-main\pothole-detection-main\project_files\yolov4_tiny.cfg"
video_path = r"C:\Users\akhal\Downloads\test.mp4"
result_path = r"C:\Users\akhal\Downloads\pothole-detection-main\pothole-detection-main\pothole_coordinates"

# Ensure necessary files exist
for file in [obj_names_path, weights_path, config_path, video_path]:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Error: '{file}' not found.")

# Ensure result directory exists
os.makedirs(result_path, exist_ok=True)

# Read class labels
with open(obj_names_path, "r") as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Load YOLO model
net1 = cv.dnn.readNet(weights_path, config_path)

# Check CUDA availability
try:
    net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    print("[INFO] Running on GPU")
except:
    net1.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    net1.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    print("[WARN] CUDA not available, switching to CPU")

model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# Open video file
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Error: Could not open video file. Check the path.")

width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(5))

# Define video writer
result = cv.VideoWriter(os.path.join(result_path, 'result.avi'),
                         cv.VideoWriter_fourcc(*'MJPG'),
                         fps, (width, height))

# Geolocation (for pothole logging)
g = geocoder.ip('me')

# Detection parameters
starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
pothole_index = 0
last_detection_time = 0

# Video processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    # Detect potholes
    detections = model1.detect(frame, Conf_threshold, NMS_threshold)

    # Ensure detection results are valid
    if detections is None or not isinstance(detections, tuple) or len(detections) != 3:
        continue  # Skip frame if detection fails

    classes, scores, boxes = detections

    # Skip frame if no detections
    if len(classes) == 0 or len(scores) == 0 or len(boxes) == 0:
        continue

    for (classid, score, box) in zip(classes.flatten(), scores.flatten(), boxes):
        label = "Pothole"
        x, y, w, h = box
        rec_area = w * h
        total_area = width * height

        # Only consider potholes within a reasonable size
        if score >= 0.7 and (rec_area / total_area) <= 0.1 and y < 600:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv.putText(frame, f"{label}: {score:.2f}", (x, y - 10),
                       cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

            # Save pothole images and coordinates every 2 seconds
            if time.time() - last_detection_time >= 2:
                pothole_img_path = os.path.join(result_path, f"pothole{pothole_index}.jpg")
                pothole_txt_path = os.path.join(result_path, f"pothole{pothole_index}.txt")

                cv.imwrite(pothole_img_path, frame)
                with open(pothole_txt_path, "w") as f:
                    f.write(str(g.latlng))

                pothole_index += 1
                last_detection_time = time.time()

    # Display FPS
    elapsed_time = time.time() - starting_time
    fps_text = f"FPS: {frame_counter / elapsed_time:.2f}"
    cv.putText(frame, fps_text, (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    # Show and save frame
    cv.imshow("Pothole Detection", frame)
    result.write(frame)

    # Handle window close event
    key = cv.waitKey(1) & 0xFF
    if key == ord("q") or cv.getWindowProperty("Pothole Detection", cv.WND_PROP_VISIBLE) < 1:
        break  # Exit loop when "q" is pressed or window is closed

# Cleanup
cap.release()
result.release()
cv.destroyAllWindows()
