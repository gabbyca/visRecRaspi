import cv2
import time
import numpy as np


camera = 'tcp://0.0.0.0:5000'
stream = cv2.VideoCapture(camera)

bandsaw =cv2.imread('bandsaw.jpg')
mill=cv2.imread('bigDrillPress.jpg')
chopsaw=cv2.imread('chopsaw.jpg')
drillPress= cv2.imread('drillPress.jpg')
sander=cv2.imread('sander.jpg')

#orb detection 
orb = cv2.ORB_create()

kp1 , des1 = orb.detectAndCompute(bandsaw, None)
kp2 , des2 = orb.detectAndCompute(mill, None)
kp3 , des3 = orb.detectAndCompute(chopsaw, None)
kp4 , des4 = orb.detectAndCompute(drillPress, None)
kp5 , des5 = orb.detectAndCompute(sander, None)

imgKp1 = cv2.drawKeypoints(bandsaw, kp1, None)
imgKp2 = cv2.drawKeypoints(mill, kp2, None)
imgKp3 = cv2.drawKeypoints(chopsaw, kp3, None)
imgKp4 = cv2.drawKeypoints(drillPress, kp4, None)
imgKp5 = cv2.drawKeypoints(sander, kp5, None)


#if stream is not opened print no stream 
if not stream.isOpened(): 
    print("no stream")
    exit()

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