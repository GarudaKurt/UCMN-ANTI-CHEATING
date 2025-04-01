import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import serial
import time
from ultralytics import YOLO

# Initialize Arduino Serial Communication
#arduino = serial.Serial('COM7', 115200, timeout=1)
time.sleep(2)

model = YOLO("C:\\Users\\letad\\Desktop\\UCMN-EE-AI\\PROJECT_FOUR (2)\\PROJECT_FOUR (2)\\PROJECT_FOUR\\yolo11n.pt")

# Load MediaPipe Face Landmarker
base_options = python.BaseOptions(
    model_asset_path=r"C:\\Users\\letad\Desktop\\UCMN-EE-AI\\PROJECT_FOUR (2)\\PROJECT_FOUR (2)\\PROJECT_FOUR\\face_landmarker.task"
)
options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, num_faces=5)
detector = vision.FaceLandmarker.create_from_options(options)

last_command = None  

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_filename = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
store_FILE = "C:\\Users\\letad\\Desktop\\UCMN-EE-AI\\PROJECT_FOUR (2)\\PROJECT_FOUR (2)\PROJECT_FOUR\\screenshots"

# Function to draw face landmarks and detect mouth open status
def draw_landmarks_on_image(rgb_image, detection_result):
    global last_command 

    face_landmarks_list = detection_result.face_landmarks
    blendshapes_list = detection_result.face_blendshapes
    annotated_image = np.copy(rgb_image)

    mouth_open_status = False
    open_mouth_count = 0
    closed_mouth_count = 0

    if face_landmarks_list:
        for idx, face_landmarks in enumerate(face_landmarks_list):
            blendshapes = blendshapes_list[idx]
            mouth_open_value = 0.0
            jaw_open_value = 0.0

            for blendshape in blendshapes:
                if blendshape.category_name == "mouthOpen":
                    mouth_open_value = blendshape.score
                if blendshape.category_name == "jawOpen":
                    jaw_open_value = blendshape.score

            print(f"Face {idx}: mouthOpen score: {mouth_open_value}, jawOpen score: {jaw_open_value}")

            if mouth_open_value > 0.3 or jaw_open_value > 0.3:
                mouth_open_status = True
                open_mouth_count += 1
            else:
                closed_mouth_count += 1

            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

    # We need to send command to Arduino only if status changes
    new_command = "on" if mouth_open_status else "off"
    if new_command != last_command:
        print(f"Sending to Arduino: {new_command}")
        #arduino.write(f"{new_command}\n".encode())
        last_command = new_command

    cv2.putText(annotated_image, f"Open Mouth: {open_mouth_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_image, f"Closed Mouth: {closed_mouth_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return annotated_image
def run():
    screenshot_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(image)

        # Process face landmarks if detected
        mouth_open_status = False
        if detection_result.face_landmarks:
            annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
            mouth_open_status = any(bs.score > 0.3 for face in detection_result.face_blendshapes for bs in face if bs.category_name in ["mouthOpen", "jawOpen"])
        else:
            annotated_frame = rgb_frame

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        results = model.predict(annotated_frame, imgsz=640, conf=0.4, classes=[67])

        phone_detected = False
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Cell Phone: {confidence:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                phone_detected = True

        if mouth_open_status:
            screenshot_filename = f"{store_FILE}\\screenshot_talk{screenshot_count}.png"
            cv2.imwrite(screenshot_filename, annotated_frame)
            print(f"Screenshot saved: {screenshot_filename}")
            screenshot_count += 1
        
        if  phone_detected:
            screenshot_filename = f"{store_FILE}\\screenshot_phone{screenshot_count}.png"
            cv2.imwrite(screenshot_filename, annotated_frame)
            print(f"Screenshot saved: {screenshot_filename}")
            screenshot_count += 1

        video_writer.write(annotated_frame)
        cv2.imshow('Face & Cell Phone Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


def main():
    run()

if __name__ == "__main__":
    main()