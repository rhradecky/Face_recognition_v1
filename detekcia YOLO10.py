import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Načítanie modelu YOLOv5
model = YOLO('yolov5s.pt')

# Načítanie kamery
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detekcia objektov
    results = model(frame)

    # Zobrazenie výsledkov v okne pomocou matplotlib
    for r in results:
        frame = r.plot()  # Aktualizovanie frame s výsledkami

    # Konvertovanie farieb z BGR na RGB pre matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Zobrazenie pomocou matplotlib
    plt.imshow(frame_rgb)
    plt.axis('off')  # Skrytie osí
    plt.show()

    # Ukončenie pomocou klávesy 'q' v matplotlib okne
    if plt.waitforbuttonpress(1) and plt.get_current_fig_manager().canvas.manager.key_press_handler_id == ord('q'):
        break

cap.release()
plt.close('all')
