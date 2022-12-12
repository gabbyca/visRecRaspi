import cv2
import numpy as np
import os


#instantialize orb detection
orb=cv2.ORB_create()
path = 'imagesSingle'


images = []
classNames =[]

list = os.listdir(path)

#for each class in the list
for cl in list:
    imgCur = cv2.imread(f'[{path}/{cl}', 0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])

#finds descriptors of each image in the list
def findDes(images):
    desList =[]
    for img in images:
        kp,des=orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

#ids img if descriptors are greater than finvVALL
def findID(img, desList):
    kp2,des2=orb.detectAndCompute(img, None)
    #brute force matcher --!sm processing
    bf = cv2.BFMatcher()
    matchList=[]
    #!!!!! will change finVal w/ testing
    finVal = -1
    try:
        for des in desList:
            matches =bf.knnMatch(des, des2, k=2)
            good =[]
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    #print(matchList)
    if len(matchList) != 0:
        if max(matchList) >= 1:
            finVal =matchList.index(max(matchList))
    return finVal

desList=findDes(images)





   