from ultralytics import YOLO 
import math 
import cv2 as cv 

model_name = 'yolov8n.pt'
source_path = r'C:\Users\alper\OneDrive\Desktop\dataset_hazirlama\mobese.mp4'

model = YOLO(model_name)
capture = cv.VideoCapture(source_path)

fps = capture.get(cv.CAP_PROP_FPS)
dt = 1 / fps if fps > 0 else 1 / 30  # FPS okunamazsa varsayılan 30

PIXEL_TO_METER = 0.05
id_dict = {}

while capture.isOpened():
    ret, frame = capture.read()

    if not ret:
        print('Video bitti veya kare yakalanamadi.')
        break  # Sonsuz döngüyü önlemek için break

    # Sadece araç sınıfları: 2=araba, 3=motosiklet, 5=otobüs, 7=kamyon
    trackeds = model.track(source=frame,
                           persist=True,
                           classes=[2, 3, 5, 7],
                           tracker='bytetrack.yaml',
                           verbose=False)

    for tracked in trackeds:
        if tracked.boxes is None:
            continue
            
        for box in tracked.boxes:
            if box.id is None:  # ID atanmamışsa atla (çökmeyi önler)
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            curr_id = int(box.id.cpu().numpy()[0])

            curr_cx = (x1 + x2) // 2
            curr_cy = (y1 + y2) // 2

            if curr_id in id_dict:
                prev_cx, prev_cy = id_dict[curr_id]

                d_pixel = math.sqrt((prev_cx - curr_cx)**2 + (prev_cy - curr_cy)**2)
                d_meter = d_pixel * PIXEL_TO_METER
                d_kilometer = (d_meter / dt) * 3.6

                cv.putText(frame, f'ID:{curr_id} Hiz: {d_kilometer:.1f} km/s', 
                           (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Kutuyu çiz
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            id_dict[curr_id] = (curr_cx, curr_cy)

    cv.imshow('Arac Hiz Tespit Sistemi', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):  # 'q' tuşuna basınca çıkış yap
        break

capture.release()
cv.destroyAllWindows()
