import cv2
import re
import time
import easyocr
from ultralytics import YOLO
from collections import Counter

vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO("best_plate.pt")

reader = easyocr.Reader(['en'], gpu=False)

VEHICLE_CLASSES = {2:"car", 3:"motorcycle", 5:"bus", 7:"truck"}

ocr_buffer = []
last_ocr_time = 0
OCR_INTERVAL = 0.7   # seconds

# ---------- PLATE NORMALIZATION ----------
def normalize_plate(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)

    fixes = {
        'O': '0',
        'I': '1',
        'Z': '2',
        'B': '8',
        'S': '5'
    }
    text = ''.join(fixes.get(c, c) for c in text)

    match = re.findall(r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}', text)
    return match[0] if match else ""

# ---------- BEST OCR RESULT ----------
def get_stable_plate(text):
    if text:
        ocr_buffer.append(text)
    if len(ocr_buffer) > 8:
        ocr_buffer.pop(0)
    if ocr_buffer:
        return Counter(ocr_buffer).most_common(1)[0][0]
    return ""

cap = cv2.VideoCapture("http://192.168.29.30:8080/video")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    vehicles = vehicle_model(frame, conf=0.3, verbose=False)[0]

    for v in vehicles.boxes:
        cls = int(v.cls)
        if cls not in VEHICLE_CLASSES:
            continue

        x1,y1,x2,y2 = map(int, v.xyxy[0])
        vehicle_crop = frame[y1:y2, x1:x2]

        plates = plate_model(vehicle_crop, conf=0.35, verbose=False)[0]

        for p in plates.boxes:
            px1,py1,px2,py2 = map(int, p.xyxy[0])
            plate = vehicle_crop[py1:py2, px1:px2]

            if plate.size == 0:
                continue

            # ---------- STRONG PREPROCESS ----------
            plate = cv2.resize(plate, None, fx=3, fy=3)
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            thresh = cv2.adaptiveThreshold(
                blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            plate_text = ""

            if time.time() - last_ocr_time > OCR_INTERVAL:
                ocr = reader.readtext(
                    thresh,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    detail=0
                )
                if ocr:
                    cleaned = normalize_plate(ocr[0])
                    if cleaned:
                        plate_text = cleaned
                        last_ocr_time = time.time()

            stable_plate = get_stable_plate(plate_text)

            # ---------- DRAW ----------
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1+px1,y1+py1),(x1+px2,y1+py2),(0,0,255),2)

            if stable_plate:
                cv2.putText(frame, stable_plate,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
                print("FINAL PLATE:", stable_plate)

    cv2.imshow("ANPR", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
