import cv2
import pytesseract
from pytesseract import Output
import imutils


# Nastavte cestu k Tesseract OCR, ak je potrebné
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_and_read_license_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Redukcia šumu a udržanie okrajov
    edged = cv2.Canny(gray, 30, 200)  # Detekcia hrán

    # Hľadanie kontúr v obrázku
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        print("No license plate detected")
        return image, None

    # Maskovanie inej oblasti mimo SPZ
    mask = cv2.bitwise_and(image, image)
    cv2.drawContours(mask, [screenCnt], -1, (0, 255, 0), 3)

    # Extrahovanie regiónu SPZ
    x, y, w, h = cv2.boundingRect(screenCnt)
    plate = gray[y:y + h, x:x + w]

    # Použitie Tesseract na rozpoznanie textu
    text = pytesseract.image_to_string(plate, config='--psm 8')
    print(f"Detected license plate Text: {text}")

    # Zobrazenie výsledkov
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)

    return image, text


# RTSP URL alebo video súbor
#rtsp_url = 'rtsp://admin:apmedia1303@172.27.40.61:554/Streaming/Channels/102'
rtsp_url = 'rtsp://admin:Apmedia1303@192.168.0.5:554/Streaming/Channels/2'

video_capture = cv2.VideoCapture(rtsp_url)
#video_capture = cv2.VideoCapture(0)  # Použitie webkamery

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame, plate_text = detect_and_read_license_plate(frame)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
