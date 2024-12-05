import numpy as np  
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import random
import cv2
from PIL import Image

model = YOLO("yolov8n.pt") 

image_add = "../YOLOv8/data_arman/phone/10.jpg"

image = cv2.imread(image_add)
h, w, c = image.shape
print(f"The image has dimensions {w}x{h} and {c} channels.")

result_predict = model.predict(source = image, imgsz=(640))

plot=result_predict[0].plot()
plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
display(Image.fromarray(plot))

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    result_predict = model.predict(source = frame, imgsz=(640))
    plot=result_predict[0].plot()
    #plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame",plot)
    if cv2.waitKey(25) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()