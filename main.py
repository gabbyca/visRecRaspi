import cv2
import time
import numpy as np
import classifier

#initialize camera
camera = 'tcp://0.0.0.0:5000'
stream = cv2.VideoCapture(camera)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def findFace(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 3)
    for(x,y,w,h) in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), color =(0,255,0), thickness=5)
    return frame

#set dimensions
fps = int(stream.get(cv2.CAP_PROP_FPS))

t= time.localtime()
current_time = time.strftime("%H:%M:%S", t)

#write video (save)
output = cv2.VideoWriter('videoStorageOpencv/' + current_time + '.avi',cv2.VideoWriter_fourcc('M','J','P','G'),fps,(70,70))

#variables from other classes
desList = classifier.desList

#if stream found, return the frames of the screen 
while(True):
    #read frames of stream
    ret, frame = stream.read()  
    imgOriginal = frame.copy()
   
    #classifier (drillpress & bandsaw detection)
    # id = classifier.findID(frame, desList)
    # if id != -1:
    #     cv2.putText(imgOriginal, classifier.classNames[id], (50,50), cv2.FONT_KERSHEY_COMPLEX,1,(0,0,255),1)

    #facialRec
    frame = findFace(frame)

    cv2.imshow('w', frame)
    output.write(frame)

    if cv2.waitKey(1) == ord('q'):
        break
    


#release objs
output.release()
stream.release()
cv2.destroyAllWindows()
