import cv2
import time
import numpy as np
import classifier
import facialRec

#initialize camera
camera = 'tcp://0.0.0.0:5000'
stream = cv2.VideoCapture(camera)

#set dimensions
fps = int(stream.get(cv2.CAP_PROP_FPS))
width = int(stream.get(3))
height = int(stream.get(4))
size = (width, height)

t= time.localtime()
current_time = time.strftime("%H:%M:%S", t)

#write video (save)
output = cv2.VideoWriter('videoStorageOpencv/' + current_time + '.avi',cv2.VideoWriter_fourcc('M','J','P','G'),fps,size)

#variables from other classes
desList = classifier.desList

#if stream found, return the frames of the screen 
while(True):
    #read frames of stream
    ret, frame = stream.read()  
    output.write(frame)
    imgOriginal = frame.copy()

    #classifier (drillpress & bandsaw detection)
    id = classifier.findID(frame, desList)
    if id != -1:
        cv2.putText(imgOriginal, classifier.classNames[id], (50,50), cv2.FONT_KERSHEY_COMPLEX,1,(0,0,255),1)

    #facialRec
    facialRec.findFaces()
   

    if cv2.waitKey(1) == ord('q'):
        break
    


#release objs
output.release()
stream.release()
cv2.destroyAllWindows()
