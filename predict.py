from ultralytics import YOLO
import numpy as np
import cv2
import time

trained_model_path = 'runs/detect/train34/weights/best.pt'
# Load the trained model
model = YOLO(trained_model_path)

# Use the model for prediction or further fine-tuning
frame_count = 0
start_time = time.time()

cam = cv2.VideoCapture(0)
while True:
    
    ret, frame = cam.read()
    result_predict = model.predict(source = frame, imgsz=(640))
    plot=result_predict[0].plot()
    frame_count += 1
    tmp_time = time.time() - start_time 
    fps = frame_count / tmp_time
    cv2.putText(plot, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    #plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame",plot)
    if cv2.waitKey(25) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()