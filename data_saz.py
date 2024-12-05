import cv2
import numpy as np
import os
import pickle
import mediapipe as mp
import matplotlib.pyplot as plt
import warnings


def make_data(link,number_of_class):
    dataset_size = 100
    cam = cv2.VideoCapture(0)
    if not os.path.exists(os.path.join(link, f"{number_of_class}")):
        os.makedirs(os.path.join(link, f"{number_of_class}"))
    while True:
        ret, frame = cam.read()
        #frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'Press"Q"(making data for {number_of_class})', (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.2, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
    
    counter = 0
    while counter < dataset_size:
        ret, frame = cam.read()
        #frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(link, f"{number_of_class}", '{}.jpg'.format(counter)), frame)
        counter += 1
    cam.release()
    cv2.destroyAllWindows()
    
    
link = "../YOLOv8/data_arman"
classes = ["phone","pencil","cup"]
for classs in classes:
    make_data(link,classs)