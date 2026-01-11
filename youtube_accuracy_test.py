import cv2
import yt_dlp
from ultralytics import YOLO
import easyocr
import re

# ---------------- CONFIG ----------------
YOUTUBE_URL = "https://www.youtube.com/watch?v=dyf5c1iX5Xc"
VEHICLE_MODEL = YOLO("yolov8n.pt")
PLATE_MODEL = YOLO("best.pt")

reader = easyocr.Reader(['en'], gpu=False)

plate_regex = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{3,4}$')

# Get YouTube video stream
ydl_opts = {"format": "best[ext=mp4]"}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(YOUTUBE_URL, download=False)
    video_url = info["url"]

cap = cv2.VideoCapture(video_url)

correct_vehicle = 0
correct_plate = 0
total = 0

vehicle_names = {2:"Car",3:"Bike",5:"Bus",7:"Truck"}

def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

print("\nPress Y = correct | N = wrong | Q = quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    vehicles = VEHICLE_MODEL(frame, conf=0.5, classes=[2,3,5,7])
    plates = PLATE_MODEL(frame, conf=0.4)

    for v in vehicles[0].boxes:
        x1,y1,x2,y2 = map(int, v.xyxy[0])
        cls = int(v.cls[0])
        vtype = vehicle_names.get(cls, "Vehicle")

        crop_plate = None
        for p in plates[0].boxes:
            px1,py1,px2,py2 = map(int, p.xyxy[0])
            if px1 > x1 and px2 < x2:
                crop_plate = frame[py1:py2, px1:px2]
                break

        plate_text = "UNKNOWN"
        if crop_plate is not None:
            ocr = reader.readtext(crop_plate, detail=0)
            if ocr:
                text = clean_text(ocr[0])
                if plate_regex.match(text):
                    plate_text = text

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"{vtype} | {plate_text}",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        cv2.imshow("ANPR Accuracy Test", frame)

        print(f"\nDetected: {vtype} | {plate_text}")
        key = cv2.waitKey(0)

        if key == ord('y'):
            correct_vehicle += 1
            if plate_text != "UNKNOWN":
                correct_plate += 1
            total += 1
            print("✔ Correct")
        elif key == ord('n'):
            total += 1
            print("❌ Wrong")
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if total > 0:
    print("\nVehicle Accuracy:", round((correct_vehicle/total)*100,2), "%")
    print("Plate Accuracy:", round((correct_plate/total)*100,2), "%")
else:
    print("No detections")
