from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Loads the video file
cap = cv2.VideoCapture("newvideo.mp4")

tracker = None
frame_count = 0   

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (800, 600))

    # YOLO detection
    results = model(frame, conf=0.5, verbose=False)[0]

    # Detecting the object
    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if confidence > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Initializing the tracker
            try:
                tracker = cv2.TrackerMIL_create()
            except:
                tracker = cv2.legacy.TrackerMIL_create()

            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            break

    # Tracking
    if tracker is not None:
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = map(int, bbox)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking the tyre", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()