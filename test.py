import cv2
import re
from ultralytics import YOLO
from collections import defaultdict, Counter
from paddleocr import PaddleOCR

# ---------- MODELS ----------
vehicle_model = YOLO("yolov8l.pt")
plate_model = YOLO("best_plate.pt")

# ✅ PADDLE OCR INIT
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# ---------- VALID VEHICLES ----------
VALID_VEHICLES = [
    "car", "motorcycle", "bus", "truck",
    "bicycle", "train"
]

# ---------- STORAGE ----------
vehicle_plates = {}
plate_buffer = defaultdict(list)
final_vehicles = {}
stored_plates = set()
seen_ids = set()

# =========================================================
# 🔥 OCR + STRICT INDIAN FILTER
# =========================================================

def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def extract_indian_plate(ocr_list):

    merged = "".join(ocr_list)
    merged = clean_text(merged)

    if len(merged) < 8:
        return ""

    for i in range(len(merged)):
        candidate = merged[i:i+10]

        if len(candidate) < 8:
            continue

        candidate = list(candidate)

        for j, ch in enumerate(candidate):

            if j < 2:
                if ch.isdigit():
                    candidate[j] = {'0':'O','1':'I','2':'Z','5':'S','8':'B'}.get(ch, ch)

            elif 2 <= j < 4:
                if ch.isalpha():
                    candidate[j] = {'O':'0','I':'1','Z':'2','S':'5','B':'8'}.get(ch, ch)

            elif j >= len(candidate) - 4:
                if ch.isalpha():
                    candidate[j] = {
                        'O':'0','I':'1','Z':'2',
                        'S':'5','B':'8','D':'0','Q':'0'
                    }.get(ch, ch)

        candidate = "".join(candidate)

        if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$', candidate):
            return candidate

    return ""


# =========================================================
# 🔥 VIDEO
# =========================================================

cap = cv2.VideoCapture("test_images/test_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

            vehicle_crop = frame[y1:y2, x1:x2]

            plates = plate_model(vehicle_crop, conf=0.3, verbose=False)[0]

            for p in plates.boxes:
                px1, py1, px2, py2 = map(int, p.xyxy[0])

                plate = vehicle_crop[py1:py2, px1:px2]

                if plate.size == 0:
                    continue

                plate = cv2.resize(plate, None, fx=3, fy=3)

                # ✅ PADDLE OCR
                result = ocr_engine.ocr(plate)

                ocr_texts = []

                if result and result[0]:
                    for line in result[0]:
                        text = line[1][0]
                        ocr_texts.append(text)

                if not ocr_texts:
                    continue

                print("RAW OCR:", ocr_texts)

                plate_text = extract_indian_plate(ocr_texts)

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

    # =========================================================
    # 🔥 STORE FINAL DATA
    # =========================================================

    previous_ids = seen_ids.copy()
    seen_ids = current_ids.copy()

    exited_ids = previous_ids - seen_ids

    for vid in exited_ids:
        if vid in plate_buffer:

            final_plate = Counter(plate_buffer[vid]).most_common(1)[0][0]

            if final_plate not in stored_plates:

                stored_plates.add(final_plate)
                final_vehicles[vid] = final_plate

                print(f"FINAL STORED → ID:{vid} Plate:{final_plate}")

                with open("results.txt", "a") as f:
                    f.write(f"ID:{vid} Plate:{final_plate}\n")

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