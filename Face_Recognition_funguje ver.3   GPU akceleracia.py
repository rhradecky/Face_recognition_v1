import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import face_recognition
import os
import socket
import numpy as np

import face_recognition

# Získanie GPU akcelerácie pre dlib
face_recognition.api.CNN_FACE_DETECTOR_MODEL = "mmod_human_face_detector.dat"




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

# Funkcia na posielanie správy cez TCP port
def send_tcp_message(message):
    server_address = ('localhost', 5555)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.connect(server_address)
            sock.sendall(message.encode('utf-8'))
        except Exception as e:
            print(f"Could not send message: {e}")
            log.info(f"Could not send message: {e} at {str(dt.datetime.now())}")

# Zlepšenie kvality obrazu
def improve_image_quality(image):
    # Zvýšenie kontrastu a jasu
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

# Cesta k uloženým fotografiám
known_faces_dir = "C:/Users/hrade/PycharmProjects/Webcam-Face-Detect/foto"

known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

if faceCascade.empty():
    print("Error loading cascade classifier")
    sys.exit()

log.basicConfig(filename='webcam.log', level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

retry_attempts = 5

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        log.info("Unable to load camera at " + str(dt.datetime.now()))
        retry_attempts -= 1
        if retry_attempts == 0:
            break
        sleep(5)
        continue

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to grab frame")
        log.info("Failed to grab frame at " + str(dt.datetime.now()))
        break

    # Zlepšenie kvality obrazu
    frame = improve_image_quality(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    # Draw a rectangle around the faces and label them
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 255, 0)  # Default color is green for unknown faces
        if name != "Unknown":
            color = (0, 0, 255)  # Change color to red for recognized faces
            message = f"rozpoznana tvar {name}"
            send_tcp_message(message)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if anterior != len(face_locations):
        anterior = len(face_locations)
        log.info("faces: " + str(len(face_locations)) + " at " + str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
