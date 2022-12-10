import cv2
import time
import numpy as np
import orbDetection

camera = 'tcp://0.0.0.0:5000'
stream = cv2.VideoCapture(camera)


img5 = orbDetection.img5
img6 = orbDetection.img6
img8 = orbDetection.img8

cv2.imshow('img5', img5)
cv2.imshow('img6', img6)
cv2.imshow('img8', img8)

fps = int(stream.get(cv2.CAP_PROP_FPS))
width = int(stream.get(3))
height = int(stream.get(4))
size = (width, height)

t= time.localtime()
current_time = time.strftime("%H:%M:%S", t)

#write video (save)
output = cv2.VideoWriter('videoStorageOpencv/' + current_time + '.avi',cv2.VideoWriter_fourcc('M','J','P','G'),fps,size)

#if stream found, return the frames of the screen 
while(True):
    ret, frame = stream.read()
    if not ret:
        print("stream ended")
        break
    
    cv2.resize(frame, (width,height))    
    output.write(frame)
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    


output.release()
stream.release()
cv2.destroyAllWindows()
print("video saved")