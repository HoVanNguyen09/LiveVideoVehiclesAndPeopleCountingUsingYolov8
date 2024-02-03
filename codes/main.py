import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
from vidgear.gears import CamGear
from tracking import *

model=YOLO('yolov8s.pt')

stream = CamGear(source="https://www.youtube.com/watch?v=9bFOCNOarrA", stream_mode = True, logging=True).start() # YouTube Video URL as input

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow("Live Stream On YouTube")
cv2.setMouseCallback("Live Stream On YouTube", RGB)

my_file = open("/Users/hovannguyen/Desktop/SaveDisk/Documents/Projects/Yolov8/LiveVideoVehiclesAndPeopleCountingUsingYolov8/labels/lables.txt", "r")
data = my_file.read()
class_list = data.split("\n")

trackingv1 =Tracking()
trackingv2 = Tracking()

area1 = [(752, 263), (414, 384), (437, 396), (772, 272)]
area2 = [(777, 275), (445, 403), (445, 416), (796, 279)]
area3 = [(404, 356) ,(546, 442), (562, 423), (424, 342)] # Yellow
area4 = [(292, 363) ,(504, 497), (573, 498), (318, 346)] # Purple

count = 0
downcar = {}
downcarcounter = []
upcar = {}
upcarcounter = []

gooutperson = {}
gooutpersoncounter = []
enterperson = {}
enterpersoncounter = []

while True:    
    frame = stream.read()   
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame,(1020,500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []
    list2 = []
    for index, row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c = class_list[d]
        if c == 'car' or c == 'truck' or c == 'bus':
            list.append([x1, y1, x2, y2])
        elif c == 'person':
            list2.append([x1, y1, x2, y2])

    """ Venhicle """ 
    bbox_idx = trackingv1.update(list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id1 = bbox
        cx=int(x3 + x4) // 2
        cy=int(y3 + y4) // 2
        
        """ Part 1: downcar """
        result = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
        if result >= 0:
            downcar[id1] = (cx, cy)
        if id1 in downcar:
            result1 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
            if result1 >= 0:
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2) 
                cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
                if downcarcounter.count(id1) == 0:
                    downcarcounter.append(id1)

##########################################################################
        """ Part 2: upcar """
        result2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
        if result2 >= 0:
            upcar[id1] = (cx, cy)
        if id1 in upcar:
            result3 = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
            if result3 >= 0:
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2) 
                cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
                if upcarcounter.count(id1) == 0:
                    upcarcounter.append(id1)

    """ Person """ 
    bbox_idx2 = trackingv2.update(list2)
    for bbox2 in bbox_idx2:
        x32, y32, x42, y42, id2 = bbox2
        cx2=int(x32 + x42) // 2
        cy2=int(y32 + y42) // 2

        """ Part 1: go out """
        result4 = cv2.pointPolygonTest(np.array(area4, np.int32), ((cx2, cy2)), False)
        if result4 >= 0:
            gooutperson[id2] = (cx2, cy2)
        if id2 in gooutperson:
            result5 = cv2.pointPolygonTest(np.array(area3, np.int32), ((cx2, cy2)), False)
            if result5 >= 0:
                cv2.circle(frame,(cx2, cy2), 4,(0, 0, 255), -1)
                cv2.rectangle(frame, (x32, y32), (x42, y42),(255, 255, 255), 2) 
                cvzone.putTextRect(frame,f'{id2}',(x32,y32),1,1)
                if gooutpersoncounter.count(id2) == 0:
                    gooutpersoncounter.append(id2)

###########################################################################
        """ Part 2: enter """
        result6 = cv2.pointPolygonTest(np.array(area3, np.int32), ((cx2, cy2)), False)
        if result6 >= 0:
            enterperson[id2] = (cx2, cy2)
        if id2 in enterperson:
            result7 = cv2.pointPolygonTest(np.array(area4, np.int32), ((cx2, cy2)), False)
            if result7 >= 0:
                cv2.circle(frame,(cx2, cy2), 4,(0, 0, 255), -1)
                cv2.rectangle(frame,(x32, y32),(x42, y42),(255, 255, 255), 2) 
                cvzone.putTextRect(frame, f'{id2}',(x32, y32),1,1)
                if enterpersoncounter.count(id2) == 0:
                    enterpersoncounter.append(id2)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.polylines(frame, [np.array(area3, np.int32)], True, (25, 255, 222), 2)
    cv2.polylines(frame, [np.array(area4, np.int32)], True, (179, 112, 212), 2)

    number_down = len(downcarcounter)  
    # print(number_down)
    number_up = len(upcarcounter)
    # print(number_up) 
    number_goout = len(gooutpersoncounter)
    number_enter = len(enterpersoncounter)
    cvzone.putTextRect(frame, f"Xe Di Xuong: {number_down}", (50,50), 1, 1)
    cvzone.putTextRect(frame, f"Xe Di Len: {number_up}",(800,50),1,1)
    cvzone.putTextRect(frame, f"Nguoi Di Ra: {number_goout}", (50,100), 1, 1)
    cvzone.putTextRect(frame, f"Nguoi Di Vao: {number_enter}",(800,100),1,1)
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
stream.release()
cv2.destroyAllWindows()