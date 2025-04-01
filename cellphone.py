import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("C:\\Users\\geral\\Desktop\\Data\\Artificial Intelligence\\PROJECT_FOUR\\yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection, filtering only for "cell phone" (COCO class index: 67)
    results = model.predict(frame, imgsz=640, conf=0.4, classes=[67])

    # Process results and draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            confidence = box.conf[0].item()  # Get confidence score

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display detection label
            label = f"Cell Phone: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display a message on the frame
            cv2.putText(frame, "Cell Phone Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show the frame
    cv2.imshow("YOLO Cell Phone Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
