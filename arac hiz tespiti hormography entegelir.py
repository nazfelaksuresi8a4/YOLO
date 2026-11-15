from ultralytics import YOLO 
import numpy as np
import math 
import cv2 as cv 

model_name = 'yolov8n.pt'
source_path = r'C:\Users\alper\OneDrive\Desktop\dataset_hazirlama\mobese.mp4'

model = YOLO(model_name)
capture = cv.VideoCapture(source_path)

fps = capture.get(cv.CAP_PROP_FPS)
dt = 1 / fps if fps > 0 else 1 / 30

# 1. ADIM: Videodaki yoldan seçilen 4 köşe noktası (Piksel koordinatları: [X, Y])
# Kendi videona göre bu 4 köşeyi (Sol-Üst, Sağ-Üst, Sağ-Alt, Sol-Alt) güncellemelisin.
SOURCE = np.array([
    [450, 300],   # Sol-Üst (uzak)
    [800, 300],   # Sağ-Üst (uzak)
    [1100, 700],  # Sağ-Alt (yakın)
    [150, 700]    # Sol-Alt (yakın)
], dtype=np.float32)

# 2. ADIM: Seçilen bu alanın gerçek dünyadaki metre cinsinden karşılığı (Genişlik: 12m, Uzunluk: 35m)
TARGET_WIDTH = 12.0   # Örn: 3 şerit x 4 metre
TARGET_HEIGHT = 35.0  # Örn: 3 kesik çizgi + boşluk

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH, 0],
    [TARGET_WIDTH, TARGET_HEIGHT],
    [0, TARGET_HEIGHT]
], dtype=np.float32)

# Perspektif dönüşüm matrisini hesapla
perspective_matrix = cv.getPerspectiveTransform(SOURCE, TARGET)

def transform_point(x, y, matrix):
    """Piksel koordinatını gerçek dünya metre koordinatına çevirir."""
    point = np.array([[[x, y]]], dtype=np.float32)
    transformed = cv.perspectiveTransform(point, matrix)
    return transformed[0][0][0], transformed[0][0][1]

id_dict = {}

while capture.isOpened():
    ret, frame = capture.read()

    if not ret:
        print('Video bitti veya kare yakalama islemi basarisiz!')
        break

    # Kalibrasyon alanını ekranda görmek için yeşil çerçeve çiz (İstersen silebilirsin)
    cv.polylines(frame, [SOURCE.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    trackeds = model.track(source=frame,
                           persist=True,
                           classes=[2, 3, 5, 7],
                           tracker='bytetrack.yaml',
                           verbose=False)

    for tracked in trackeds:
        if tracked.boxes is None:
            continue

        for box in tracked.boxes:
            if box.id is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            curr_id = int(box.id.cpu().numpy()[0])

            # Aracın yol zeminine bastığı alt-orta noktayı almak perspektifte daha doğrudur
            curr_cx = (x1 + x2) // 2
            curr_cy = y2 

            # Piksel noktasını doğrudan METRE koordinatına dönüştür
            real_x, real_y = transform_point(curr_cx, curr_cy, perspective_matrix)

            if curr_id in id_dict:
                prev_real_x, prev_real_y = id_dict[curr_id]

                # Artık doğrudan METRE cinsinden Öklid mesafesi hesaplıyoruz
                d_meter = math.sqrt((prev_real_x - real_x)**2 + (prev_real_y - real_y)**2)
                d_kilometer = (d_meter / dt) * 3.6

                cv.putText(frame, f'arac hizi: {d_kilometer:.1f} km/s', 
                           (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            id_dict[curr_id] = (real_x, real_y)

    cv.imshow('arac hiz tespit sistemi', frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
