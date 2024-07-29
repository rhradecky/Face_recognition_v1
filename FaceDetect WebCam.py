import cv2

# Načítanie predtrénovaného modelu Haarcascade na detekciu tvárí
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Otvorenie streamu z webkamery (predvolené zariadenie je obvykle index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    # Načítanie snímku z webkamery
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot grab frame")
        break

    # Konverzia snímku do šedého tónu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detekcia tvárí v snímku
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Kreslenie obdĺžnikov okolo detegovaných tvárí
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Zobrazenie snímku
    cv2.imshow('Face Detection from Webcam', frame)

    # Ukončenie cyklu stlačením klávesy 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Uvoľnenie zdrojov
cap.release()
cv2.destroyAllWindows()
