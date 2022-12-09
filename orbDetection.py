import cv2

bandsaw =cv2.imread('bandsaw.jpg', 0)
mill=cv2.imread('bigDrillPress.jpg', 0)
drillPress= cv2.imread('drillPress.jpg', 0)

#query images 
sanderBandsawChopsaw = cv2.imread('sanderBandsawChopsaw.jpg', 0)
sanderDrillBandsaw = cv2.imread('sanderDrillBandsaw.jpg', 0)
millChopsaw = cv2.imread('millChopsaw.jpg', 0)

#orb detection 
orb = cv2.ORB_create()

#bandsaw
kp1 , des1 = orb.detectAndCompute(bandsaw, None)
kp2 , des2 = orb.detectAndCompute(sanderBandsawChopsaw, None)
kp3, des3 = orb.detectAndCompute(sanderDrillBandsaw, None)
#mill
kp4 , des4 = orb.detectAndCompute(mill, None)
kp5, des5 = orb.detectAndCompute( millChopsaw, None)
#drillPress
kp6 , des6 = orb.detectAndCompute(drillPress, None)
kp7, des7 = orb.detectAndCompute(sanderDrillBandsaw, None)

#brute force matcher
bf = cv2.BFMatcher()

mb1 = bf.knnMatch(des1, des2, k=2)
mb2 = bf.knnMatch(des1, des3, k=2)

mm1 = bf.knnMatch(des4, des5, k=2)

md1 = bf.knnMatch(des6, des7, k=2)

good = []

#may have to change 0.75
for m,n in mb1:
    if m.distance < 0.75*n.distance:
        good.append([m])
for m,n in mb2:
    if m.distance < 0.75*n.distance:
        good.append([m])

for m,n in mm1:
    if m.distance < 0.75*n.distance:
        good.append([m])

for m,n in md1:
    if m.distance < 0.75*n.distance:
        good.append([m])


img5 = cv2.drawMatchesKnn(bandsaw, kp1, sanderBandsawChopsaw, kp2, good, None, flags=2)
img6 = cv2.drawMatchesKnn(bandsaw, kp1, sanderDrillBandsaw, kp3, good, None, flags=2)
img7 = cv2.drawMatchesKnn(mill, kp4, millChopsaw, kp5, good, None, flags=2)
img8 = cv2.drawMatchesKnn(drillPress, kp6, sanderDrillBandsaw, kp7, good, None, flags=2)
