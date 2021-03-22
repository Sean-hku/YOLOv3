import os
import cv2
import numpy as np

video_name = "all2_Trim.mp4"
cap = cv2.VideoCapture(video_name)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("_processed.avi", fourcc, 25, (1920, 1080))
height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

while True:
    ret, frame = cap.read()
    if ret:
        save = frame[0: 1080, 0: 1920]
        out.write(save)








