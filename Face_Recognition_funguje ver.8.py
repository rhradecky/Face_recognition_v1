import cv2
import face_recognition
import os
import logging as log
import datetime as dt
from time import sleep
from ultralytics import YOLO
import time

# Načítanie a zakódovanie uložených fotografií
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                face_encoding = face_encodings[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(os.path.splitext(filename)[0])
            else:
                print(f"No face found in {filename}")
                log.info(f"No face found in {filename} at {str(dt.datetime.now())}")

    return known_face_encodings, known_face_names

# Cesta k uloženým fotografiám
known_faces_dir = "C:/Users/hrade/PycharmProjects/Webcam-Face-Detect/foto"
known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

# Načítanie modelu YOLOv5
model = YOLO("yolov5s.pt")

log.basicConfig(filename='webcam.log', level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

retry_attempts = 5
fps_limit = 5  # Limit FPS
prev_time = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        log.info("Unable to load camera at " + str(dt.datetime.now()))
        retry_attempts -= 1
        if retry_attempts == 0:
            break
        sleep(5)
        continue

    ret, frame = video_capture.read()

    if not ret:
        print("Failed to grab frame")
        log.info("Failed to grab frame at " + str(dt.datetime.now()))
        break

    # Limitovanie FPS
    current_time = time.time()
    if (current_time - prev_time) < 1.0 / fps_limit:
        continue
    prev_time = current_time

    # Zníženie rozlíšenia snímky na rýchlejšie spracovanie
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Detekcia objektov pomocou YOLOv5
    yolo_results = model(small_frame)

    # Rozpoznávanie tvárí
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = min(range(len(face_distances)), key=face_distances.__getitem__)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    # Kreslenie obrysov a výpis názvov na originálnu snímku
    for result in yolo_results:
        boxes = result.boxes
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0] * 2)  # Obnovenie rozlíšenia
            score = box.conf[0]
            label = int(box.cls[0])
            class_name = model.names[label]

            if score > 0.5:  # Prah na filtrovanie detekcií
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(frame, f'{class_name}: {score:.2f}', (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2  # Obnovenie rozlíšenia
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if anterior != len(face_locations):
        anterior = len(face_locations)
        log.info("faces: " + str(len(face_locations)) + " at " + str(dt.datetime.now()))

    # Zobrazenie snímky
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
