# YOLO TRACK - NESNE TAKİP SİSTEMİ

# Gerekli kütüphaneler
from ultralytics import YOLO
import cv2

# YOLO modelini yüklüyoruz
model = YOLO("yolov8n.pt")

# Bilgisayar kamerasını açıyoruz
kamera = cv2.VideoCapture(0)

# Kamera açıldı mı kontrol ediyoruz
if not kamera.isOpened():
    print("Kamera açılamadı!")
    exit()

# Kamera açık olduğu sürece çalışır
while True:

    # Kameradan görüntü alıyoruz
    basarili, kare = kamera.read()

    # Görüntü alınamazsa döngüyü bitiriyoruz
    if not basarili:
        break

    # YOLO TRACK FONKSİYONU
    sonuclar = model.track(
        source=kare,          # İşlenecek görüntü
        persist=True,         # Önceki takip bilgilerini korur
        tracker="bytetrack.yaml",  # Takip algoritması
        conf=0.5,              # Güven eşiği
        verbose=False         # Konsol mesajlarını kapatır
    )

    # Algılama sonuçlarını inceliyoruz
    for sonuc in sonuclar:

        kutular = sonuc.boxes

        # Nesne bulunup bulunmadığını kontrol ediyoruz
        if kutular is None:
            continue

        # Takip ID'lerini alıyoruz
        if kutular.id is not None:
            idler = kutular.id.int().cpu().tolist()
        else:
            idler = [None] * len(kutular)

        # Her nesneyi ayrı ayrı inceliyoruz
        for kutu, takip_id in zip(kutular, idler):

            # Nesnenin koordinatlarını alıyoruz
            x1, y1, x2, y2 = kutu.xyxy[0].int().cpu().tolist()

            # Nesnenin sınıf numarasını alıyoruz
            sinif_id = int(kutu.cls[0])

            # Sınıf numarasını nesne adına dönüştürüyoruz
            nesne_adi = model.names[sinif_id]

            # Nesnenin güven oranını alıyoruz
            guven = float(kutu.conf[0])

            # Nesnenin etrafına yeşil kutu çiziyoruz
            cv2.rectangle(
                kare,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Nesnenin bilgilerini hazırlıyoruz
            etiket = f"{nesne_adi} ID:{takip_id} Guven:{guven:.2f}"

            # Bilgileri görüntünün üzerine yazıyoruz
            cv2.putText(
                kare,
                etiket,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # İşlenmiş görüntüyü ekranda gösteriyoruz
    cv2.imshow("YOLO NESNE TAKIP", kare)

    # ESC tuşuna basılırsa programı kapatıyoruz
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Kamerayı serbest bırakıyoruz
kamera.release()

# Açık pencereleri kapatıyoruz
cv2.destroyAllWindows()

print("Takip sistemi kapatıldı!")