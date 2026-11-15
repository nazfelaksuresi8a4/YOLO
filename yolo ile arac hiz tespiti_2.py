from ultralytics import YOLO 
import math 
import cv2 as cv 

model_name = 'yolov8n.pt'
source_path = r'C:\Users\alper\OneDrive\Desktop\dataset_hazirlama\mobese.mp4'

model = YOLO(model_name)
capture = cv.VideoCapture(source_path)

fps = capture.get(cv.CAP_PROP_FPS)
dt = 1 / fps

PIXEl_TO_METER = 0.05   #kalibre edilebilir bir değer

id_dict = {}

while capture.isOpened():
    ret,frame = capture.read()

    if not ret:
        print('kare yakalama islemi basarisiz!')
        continue

    trackeds = model.track(source=frame,
                           persist=True,
                           tracker='bytetrack.yaml')

    for tracked in trackeds:
        for box in tracked.boxes:
            x1,y1,x2,y2 = map(int,box.cpu().numpy().xyxy[0])
            curr_id = int(box.cpu().numpy().id)

            if curr_id in id_dict:
                prev_x1,prev_y1,prev_x2,prev_y2 = id_dict[curr_id]

                curr_cx = (x1 + x2) // 2
                curr_cy = (y1 + y2) // 2

                prev_cx = (prev_x1 + prev_x2) // 2
                prev_cy = (prev_y1 + prev_y2) // 2

                d_pixel = math.sqrt((prev_cx - curr_cx)**2 + (prev_cy - curr_cy)**2)    #pixel birimimden uzaklık
                d_meter = d_pixel * PIXEl_TO_METER

                d_kilometer = (d_meter / dt) * 3.6

                print(d_kilometer)

                frame = cv.putText(frame,f'arac hizi: {d_kilometer:.1f}',(curr_cx,curr_cy),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)

            id_dict[curr_id] = (x1,y1,x2,y2)

    cv.imshow('arac hiz tespit sistemi',frame)
    cv.waitKey(25)
