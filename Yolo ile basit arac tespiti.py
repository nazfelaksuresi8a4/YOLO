
from ultralytics import YOLO
import cv2

# YOLO modelini yükle
model = YOLO("yolov8n.pt")  #modeli yükle 

frame = cv2.imread(r"C:\Users\alper\Downloads\German-Autobahn.jpg")  #gmrseli yükle

results = model(frame)    #modele görseli ver

names = model.names    #modelin etiketlerini al
labels = {names[key] : key for key in names}    #etiketleri ters çevir

boxes = results[0].boxes    #modele verilen görselden modelin yaptığı analiz sonuçlarını al ve ilk analiz sonucundaki tespit edilen kutu tensorlerini çek

for box in boxes:      #kutu tensörlerini for dönüsüne ver her tensmr box'a atansın
    if box.cls[0] == labels['car']:    #eğer bulunan kutunun sınıfı car sınıfı ise işleme devam et
        rects = box.xyxy[0]  #kutunun x ve y başlangıç-bitiş koordinatlarını al

        x1,y1,x2,y2 = map(int,rects)    #n=4 uzunlugundaki x1,y1,x2,y2 olarak koordinat bilgisi içeren ondalık sayılı vektöre map ile her bir elemanına int uygula ve tam sayıya çevirip unpack et

        print(x1,y1,x2,y2)    #bulunan koordinatları yazdır

        frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)    #kutuları çiz   


cv2.imshow('scanned cars',frame)    #görseli göster
cv2.waitKey(0)    #esc tuşuna basılmasın bekle
