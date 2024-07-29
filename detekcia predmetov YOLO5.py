import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Načítanie predtrénovaného modelu YOLOv5
model = YOLO("yolov5s.pt")

rtsp_url = 'rtsp://admin:Apmedia1303@192.168.0.5:554/1'
# cap = cv2.VideoCapture(rtsp_url)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open video stream")
    exit()

# Zoznam názvov tried (v závislosti na vašom modelovom trénovaní, toto je pre COCO dataset)
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detekcia objektov pomocou YOLOv5
    results = model(frame)

    # Kreslenie obrysov a výpis názvov na rám
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            score = box.conf[0]
            label = int(box.cls[0])
            class_name = class_names[label]

            if score > 0.5:  # Prah na filtrovanie detekcií
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(frame, f'{class_name}: {score:.2f}', (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Konvertovanie farieb z BGR na RGB pre matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Zobrazenie pomocou matplotlib
    plt.imshow(frame_rgb)
    plt.axis('off')  # Skrytie osí
    plt.pause(0.001)  # Krátka pauza pre aktualizáciu obrázku

    # Ukončenie pomocou klávesy 'q' v matplotlib okne
    if plt.waitforbuttonpress(1) and plt.get_current_fig_manager().canvas.manager.key_press_handler_id == ord('q'):
        break

cap.release()
plt.close('all')
