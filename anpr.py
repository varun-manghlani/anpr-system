import cv2
import re
import easyocr
from ultralytics import YOLO
from collections import defaultdict, Counter

# ---------- MODELS ----------
vehicle_model = YOLO("yolov8l.pt")
plate_model = YOLO("best_plate.pt")

reader = easyocr.Reader(['en'], gpu=False)

# ---------- VALID VEHICLES ----------
VALID_VEHICLES = [
    "car", "motorcycle", "bus", "truck",
    "bicycle", "train"
]

# =========================================================
# 🔥 OCR + STRICT INDIAN FILTER
# =========================================================

def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def extract_indian_plate(ocr_list):

    # 🔥 STEP 1: MERGE OCR TEXT
    merged = "".join(ocr_list)
    merged = clean_text(merged)

    if len(merged) < 8:
        return ""

    # 🔥 STEP 2: SLIDING WINDOW
    for i in range(len(merged)):
        candidate = merged[i:i+10]

        if len(candidate) < 8:
            continue

        candidate = list(candidate)

        # 🔥 STEP 3: POSITION-BASED CORRECTION
        for j, ch in enumerate(candidate):

            # 0–1 → LETTERS
            if j < 2:
                if ch.isdigit():
                    candidate[j] = {'0':'O','1':'I','2':'Z','5':'S','8':'B'}.get(ch, ch)

            # 2–3 → DIGITS
            elif 2 <= j < 4:
                if ch.isalpha():
                    candidate[j] = {'O':'0','I':'1','Z':'2','S':'5','B':'8'}.get(ch, ch)

            # LAST 4 → DIGITS
            elif j >= len(candidate) - 4:
                if ch.isalpha():
                    candidate[j] = {
                        'O':'0','I':'1','Z':'2',
                        'S':'5','B':'8','D':'0','Q':'0'
                    }.get(ch, ch)

        candidate = "".join(candidate)

        # 🔥 FINAL STRICT VALIDATION
        if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$', candidate):
            return candidate

    return ""


# =========================================================
# 🔥 VIDEO
# =========================================================

cap = cv2.VideoCapture("test_images/test_video.mp4")
# cap = cv2.VideoCapture(0)

vehicle_plates = {}
plate_buffer = defaultdict(list)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------- TRACKING ----------
    results = vehicle_model.track(
        frame,
        persist=True,
        conf=0.15,
        iou=0.5,
        imgsz=960,
        verbose=False
    )[0]

    current_ids = set()

    if results.boxes is not None:

        for box in results.boxes:

            if box.id is None:
                continue

            track_id = int(box.id)
            cls = int(box.cls)

            vehicle_type = vehicle_model.names[cls]

            if vehicle_type not in VALID_VEHICLES:
                continue

            vehicle_type = vehicle_type.upper()

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            current_ids.add(track_id)

            # ---------- VEHICLE CROP ----------
            vehicle_crop = frame[y1:y2, x1:x2]

            # ---------- PLATE DETECTION ----------
            plates = plate_model(vehicle_crop, conf=0.3, verbose=False)[0]

            for p in plates.boxes:
                px1, py1, px2, py2 = map(int, p.xyxy[0])

                plate = vehicle_crop[py1:py2, px1:px2]

                if plate.size == 0:
                    continue

                # ---------- OCR PREPROCESS ----------
                plate = cv2.resize(plate, None, fx=3, fy=3)
                gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

                ocr = reader.readtext(
                    thresh,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    detail=0
                )

                if not ocr:
                    continue

                print("RAW OCR:", ocr)

                plate_text = extract_indian_plate(ocr)

                if not plate_text:
                    continue

                # ---------- STABLE OCR ----------
                plate_buffer[track_id].append(plate_text)

                if len(plate_buffer[track_id]) > 10:
                    plate_buffer[track_id].pop(0)

                stable_plate = Counter(
                    plate_buffer[track_id]
                ).most_common(1)[0][0]

                vehicle_plates[track_id] = stable_plate

                # DRAW PLATE BOX
                cv2.rectangle(frame,
                              (x1+px1, y1+py1),
                              (x1+px2, y1+py2),
                              (0, 0, 255), 2)

                break

            # ---------- LABEL ----------
            label = f"{vehicle_type} ID:{track_id}"

            if track_id in vehicle_plates:
                label += f" {vehicle_plates[track_id]}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            cv2.putText(frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,0),
                        2)

    # ---------- LIVE COUNT ----------
    cv2.putText(frame,
                f"Live Vehicles: {len(current_ids)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                3)

    cv2.imshow("FINAL INDIAN ANPR SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()