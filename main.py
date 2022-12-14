import cv2
import time
import numpy as np
import os
import face_recognition


#initialize camera
camera = 'tcp://192.168.1.252:5000'
stream = cv2.VideoCapture(camera)


# path = "extractedFaces/"
# knownFaceEncodings = []
# images = os.listdir(path)
# for _ in images:
#     gray = cv2.cvtColor(_, cv2.COLOR_BGR2GRAY)
#     face = faceCascade.detectMultiScale(gray, 1.3, 6)
#     for(x,y,w,h) in face:
#             frame = cv2.rectangle(frame, (x,y), (x+w, y+h), color =(0,255,0), thickness=5)
#             face = frame[y : y+h, x : x+w]


def findFace(frame):
    #need to figure out stopwatch
    i=0 
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 6)
    for(x,y,w,h) in faces:
            file = open("logs.txt","w")
            seconds = time.time()
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), color =(0,255,0), thickness=5)
            cv2.imwrite('databaseImages/student' + str(i) + '.jpg', frame)
        
            file.write("student " + str(i) + "was in lab for "  + "minutes")
            file.close()
    i+=1
    return frame 




#set dimensions
fps = int(stream.get(cv2.CAP_PROP_FPS))

t= time.localtime()
current_time = time.strftime("%H:%M:%S", t)

#write video (save)
output = cv2.VideoWriter('videoStorageOpencv/' + current_time + '.avi',cv2.VideoWriter_fourcc('M','J','P','G'),fps,(70,70))

#if stream found, return the frames of the screen 
while(True):
    #read frames of stream
    ret, frame = stream.read()  
    imgOriginal = frame.copy()
   
    #color finder
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lowerBlue = np.array([110,50,50])
    # upperBlue = np.array([130,225,255])
    # mask = cv2.inRange(hsv, lowerBlue, upperBlue)
    # for (x,y,w,h) in mask:
    #     cv2.rectangle(frame, (x,y), (x+w, y+h), color =(0,0,255), thickness=5)
    # result = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow('frame', result)
   

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
