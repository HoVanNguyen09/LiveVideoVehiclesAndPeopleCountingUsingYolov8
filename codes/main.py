import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
from vidgear.gears import CamGear
from tracking import *
from function import track_count

""" Load Model """
model=YOLO('yolov8s.pt')

""" Get live video on YouTube """
stream = CamGear(source='https://www.youtube.com/watch?v=9bFOCNOarrA', stream_mode = True, logging=True).start()

def RGB(event, x, y):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)
cv2.setMouseCallback("Live Stream On YouTube", RGB)

""" Read all label name """
my_file = open("/Users/hovannguyen/Desktop/SaveDisk/Documents/Projects/Yolov8/LiveVideoVehiclesAndPeopleCountingUsingYolov8/labels/lables.txt", "r")
data = my_file.read()
class_list = data.split("\n")

""" Variable to use """
count = 0
tracking_v1 = Tracking()
tracking_v2 = Tracking()

""" Pixel coordinates to draw the frame """
area1 = [(752, 263), (414, 384), (437, 396), (772, 272)] # Green
area2 = [(777, 275), (445, 403), (445, 416), (796, 279)] # Blue
area3 = [(404, 356) ,(546, 442), (562, 423), (424, 342)] # Yellow
area4 = [(292, 363) ,(504, 497), (573, 498), (318, 346)] # Purple

""" Using CV library to Demo """
while True:    
    frame = stream.read()   
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame,(1020,500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list_vehicle = []
    list_person = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d=int(row[5])
        c = class_list[d]
        if c == "car" or c == "truck" or c == "bus":
            list_vehicle.append([x1, y1, x2, y2])
        elif c == "person":
            list_person.append([x1, y1, x2, y2])

    down_car_count, up_car_counter = track_count(tracking_v1, list_vehicle, area1, area2, frame)
    go_out_person_counter, enter_person_counter = track_count(tracking_v2, list_person, area3, area4, frame, type = False)

    """ Draw frame """
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.polylines(frame, [np.array(area3, np.int32)], True, (25, 255, 222), 2)
    cv2.polylines(frame, [np.array(area4, np.int32)], True, (179, 112, 212), 2)

    """ Counting """
    number_down = len(down_car_count)  
    number_up = len(up_car_counter)
    number_goout = len(go_out_person_counter)
    number_enter = len(enter_person_counter)

    """ Draw frame to show result """
    cvzone.putTextRect(frame, f"Xe Di Xuong: {number_down}", (50,50), 1, 1)
    cvzone.putTextRect(frame, f"Xe Di Len: {number_up}",(800,50),1,1)
    cvzone.putTextRect(frame, f"Nguoi Di Ra: {number_goout}", (50,100), 1, 1)
    cvzone.putTextRect(frame, f"Nguoi Di Vao: {number_enter}",(800,100),1,1)
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()