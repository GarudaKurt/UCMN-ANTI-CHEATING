import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import serial
import time

arduino = serial.Serial('COM7', 115200, timeout=1)
time.sleep(2)

base_options = python.BaseOptions(model_asset_path='C:\\Users\\geral\\Desktop\\PROJECT_FOUR\\face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True)
detector = vision.FaceLandmarker.create_from_options(options)

last_command = None  

def draw_landmarks_on_image(rgb_image, detection_result):
    global last_command 

    face_landmarks_list = detection_result.face_landmarks
    blendshapes_list = detection_result.face_blendshapes
    annotated_image = np.copy(rgb_image)

    mouth_open_text = ""
    mouth_open_status = False

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        blendshapes = blendshapes_list[idx]

        mouth_open_value = 0.0
        jaw_open_value = 0.0

        for blendshape in blendshapes:
            if blendshape.category_name == "mouthOpen":
                mouth_open_value = blendshape.score
            if blendshape.category_name == "jawOpen":
                jaw_open_value = blendshape.score

        print(f"mouthOpen score: {mouth_open_value}, jawOpen score: {jaw_open_value}")

        if mouth_open_value > 0.3 or jaw_open_value > 0.3:
            mouth_open_text = "Mouth Open"
            mouth_open_status = True
        else:
            mouth_open_text = "Mouth Closed"
            mouth_open_status = False

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

        cv2.putText(annotated_image, mouth_open_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    new_command = "on" if mouth_open_status else "off"
    if new_command != last_command:
        print(f"Sending: {new_command}")
        arduino.write(f"{new_command}\n".encode())
        last_command = new_command

    return annotated_image


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(image)
    annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('Face Landmarks and Mouth Open Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
