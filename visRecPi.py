import cv2
import time
import numpy as np


camera = 'tcp://0.0.0.0:5000'
stream = cv2.VideoCapture(camera)

bandsaw =cv2.imread('bandsaw.jpg', 0)
mill=cv2.imread('bigDrillPress.jpg', 0)
chopsaw=cv2.imread('chopsaw.jpg', 0)
drillPress= cv2.imread('drillPress.jpg', 0)


#orb detection 
orb = cv2.ORB_create()

kp1 , des1 = orb.detectAndCompute(bandsaw, None)
kp2 , des2 = orb.detectAndCompute(mill, None)
kp3 , des3 = orb.detectAndCompute(chopsaw, None)
kp4 , des4 = orb.detectAndCompute(drillPress, None)


#brute force matcher
bf = cv2.BFMatcher()
matchesBandsaw1 = bf.knnMatch(des1, des2, k=2)
matchesBandsaw2 = bf.knnMatch(des1, des3, k=2)
matchesBandsaw3 = bf.knnMatch(des1, des4, k=2)

matchesMill1 = bf.knnMatch(des2, des1, k=2)
matchesMill2 = bf.knnMatch(des2, des3, k=2)
matchesMill3 = bf.knnMatch(des2, des4, k=2)

matchesChopsaw1 = bf.knnMatch(des3, des1, k=2)
matchesChopsaw2 = bf.knnMatch(des3, des2, k=2)
matchesChopsaw3 = bf.knnMatch(des3, des4, k=2)

matchesdrillPress1 = bf.knnMatch(des4, des1, k=2)
matchesdrillPress2 = bf.knnMatch(des4, des2, k=2)
matchesdrillPress3 = bf.knnMatch(des4, des3, k=2)

good = []
#may have to change 0.75
for m,n in matchesBandsaw1:
    if m.distance < 0.75*n.distance:
        good.append([m])
for m,n in matchesBandsaw2:
    if m.distance < 0.75*n.distance:
        good.append([m])
for m,n in matchesBandsaw3:
    if m.distance < 0.75*n.distance:
        good.append([m])

for m,n in matchesMill1:
    if m.distance < 0.75*n.distance:
        good.append([m])
for m,n in matchesMill2:
    if m.distance < 0.75*n.distance:
        good.append([m])
for m,n in matchesMill3:
    if m.distance < 0.75*n.distance:
        good.append([m])

for m,n in matchesChopsaw1:
    if m.distance < 0.75*n.distance:
        good.append([m])
for m,n in matchesChopsaw2:
    if m.distance < 0.75*n.distance:
        good.append([m])
for m,n in matchesChopsaw3:
    if m.distance < 0.75*n.distance:
        good.append([m])

for m,n in matchesdrillPress1:
    if m.distance < 0.75*n.distance:
        good.append([m])
for m,n in matchesdrillPress2:
    if m.distance < 0.75*n.distance:
        good.append([m])
for m,n in matchesdrillPress3:
    if m.distance < 0.75*n.distance:
        good.append([m])



img5 = cv2.drawMatchesKnn(bandsaw, kp1, mill, kp2, good, None, flags=2)
img6 = cv2.drawMatchesKnn(bandsaw, kp1, chopsaw, kp3, good, None, flags=2)
img7 = cv2.drawMatchesKnn(bandsaw, kp1, drillPress, kp4, good, None, flags=2)

img8 = cv2.drawMatchesKnn(mill, kp2, bandsaw, kp1, good, None, flags=2)
img9 = cv2.drawMatchesKnn(mill, kp2, chopsaw, kp3, good, None, flags=2)
img10 = cv2.drawMatchesKnn(mill, kp2, drillPress, kp4, good, None, flags=2)

img11 = cv2.drawMatchesKnn(chopsaw, kp3, bandsaw, kp1, good, None, flags=2)
img12 = cv2.drawMatchesKnn(chopsaw, kp3, mill, kp2, good, None, flags=2)
img13 = cv2.drawMatchesKnn(chopsaw, kp3, drillPress, kp4, good, None, flags=2)

img14 = cv2.drawMatchesKnn(drillPress, kp4, bandsaw, kp1, good, None, flags=2)
img15 = cv2.drawMatchesKnn(drillPress, kp4, mill, kp2, good, None, flags=2)
img16 = cv2.drawMatchesKnn(drillPress, kp4, chopsaw, kp3, good, None, flags=2)

cv2.imshow('img5', img5)






# imgKp1 = cv2.drawKeypoints(bandsaw, kp1, None)
# imgKp2 = cv2.drawKeypoints(mill, kp2, None)
# imgKp3 = cv2.drawKeypoints(chopsaw, kp3, None)
# imgKp4 = cv2.drawKeypoints(drillPress, kp4, None)
# imgKp5 = cv2.drawKeypoints(sander, kp5, None)

# cv2.imshow('kp1', imgKp1)
# cv2.imshow('kp2', imgKp2)
# cv2.imshow('kp3', imgKp3)
# cv2.imshow('kp4', imgKp4)
# cv2.imshow('kp5', imgKp5)

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