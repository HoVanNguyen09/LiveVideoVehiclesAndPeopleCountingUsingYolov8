import cv2
import numpy as np
import cvzone

""" Variable to use """
down_vehicle = {}
downcouter_vehicle = []
up_vehicle = {}
upcounter_vehicle = []

down_person = {}
downcouter_person = []
up_person = {}
upcounter_person = []

""" Function """
def track_count(tracking, list, area1, area2, frame, type = True):
    ################    type = True  ####################
    if type == True:
        bbox_idx = tracking.update(list)
        for bbox in bbox_idx:
            xbb_1, ybb_1, xbb_2, ybb_2, id = bbox
            cx = int(xbb_1 + xbb_2) // 2
            cy = int(ybb_1 + ybb_2) // 2
            
            """ Part 1: down """
            result_1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
            if result_1 >= 0:
                down_vehicle[id] = (cx, cy)
            if id in down_vehicle:
                result_1_2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
                if result_1_2 >= 0:
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.rectangle(frame,(xbb_1, ybb_1),(xbb_2, ybb_2),(255,255,255),2) 
                    cvzone.putTextRect(frame,f'{id}',(xbb_1, ybb_1),1,1)
                    if downcouter_vehicle.count(id) == 0:
                        downcouter_vehicle.append(id)

            """ Part 2: up """
            result_2_1 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
            if result_2_1 >= 0:
                up_vehicle[id] = (cx, cy)
            if id in up_vehicle:
                result_2_2 = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
                if result_2_2 >= 0:
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.rectangle(frame,(xbb_1, ybb_1),(xbb_2, ybb_2),(255,255,255),2) 
                    cvzone.putTextRect(frame,f'{id}',(xbb_1, ybb_1),1,1)
                    if upcounter_vehicle.count(id) == 0:
                        upcounter_vehicle.append(id)
        return downcouter_vehicle, upcounter_vehicle
    
    ################    type = False  ####################
    else:
        bbox_idx = tracking.update(list)
        for bbox in bbox_idx:
            xbb_1, ybb_1, xbb_2, ybb_2, id_1 = bbox
            cx = int(xbb_1 + xbb_2) // 2
            cy = int(ybb_1 + ybb_2) // 2
            
            """ Part 1: down """
            result_1 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
            if result_1 >= 0:
                down_person[id_1] = (cx, cy)
            if id_1 in down_person:
                result_1_2 = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
                if result_1_2 >= 0:
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.rectangle(frame,(xbb_1, ybb_1),(xbb_2, ybb_2),(255,255,255),2) 
                    cvzone.putTextRect(frame,f'{id_1}',(xbb_1, ybb_1),1,1)
                    if downcouter_person.count(id_1) == 0:
                        downcouter_person.append(id_1)

            """ Part 2: up """
            result_2_1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
            if result_2_1 >= 0:
                up_person[id_1] = (cx, cy)
            if id_1 in up_person:
                result_2_2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
                if result_2_2 >= 0:
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.rectangle(frame,(xbb_1, ybb_1),(xbb_2, ybb_2),(255,255,255),2) 
                    cvzone.putTextRect(frame,f'{id_1}',(xbb_1, ybb_1),1,1)
                    if upcounter_person.count(id_1) == 0:
                        upcounter_person.append(id_1)
        return downcouter_person, upcounter_person