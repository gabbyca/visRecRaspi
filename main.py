import cv2
import time
import numpy as np
import classifier

camera = 'tcp://0.0.0.0:5000'
stream = cv2.VideoCapture(camera)


fps = int(stream.get(cv2.CAP_PROP_FPS))
width = int(stream.get(3))
height = int(stream.get(4))
size = (width, height)

t= time.localtime()
current_time = time.strftime("%H:%M:%S", t)

#write video (save)
output = cv2.VideoWriter('videoStorageOpencv/' + current_time + '.avi',cv2.VideoWriter_fourcc('M','J','P','G'),fps,size)

desList = classifier.desList

#if stream found, return the frames of the screen 
while(True):
    ret, frame = stream.read()  
    output.write(frame)
    imgOriginal = frame.copy()
  
    id = classifier.findID(frame, desList)
    if id != -1:
        cv2.putText(imgOriginal, classifier.classNames[id], (50,50), cv2.FONT_KERSHEY_COMPLEX,1,(0,0,255),1)

    if cv2.waitKey(1) == ord('q'):
        break
    


output.release()
stream.release()
cv2.destroyAllWindows()
