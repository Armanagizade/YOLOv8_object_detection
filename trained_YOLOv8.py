import numpy as np  
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

import random
import cv2
from PIL import Image

Final_model = YOLO('yolov8n.pt')

Result_Final_model = Final_model.train(data='datasets/data_arman/data.yaml',epochs = 30, batch = -1, optimizer = 'auto')

