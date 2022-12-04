import cv2
import matplotlib

stream = cv2.VideoCapture(0)
grab, frame = stream.read()
matplotlib.pyplot.imshow(frame)

grab, frame = stream.read()
matplotlib.pyplot.imshow(frame)






#-------------------------------------------------
#acess video, webcam 
#stream = cv2.VideoCapture(0)

#if stream is not opened print no stream 
#if not stream.isOpened(): 
    #print("no stream")
    #exit()

#if stream found, return the frames of the screen 
#while(True):
    #ret, frame = stream.read()
    #cv2.imshow("Webcam", frame)
    #if cv2.waitKey(1) == ord('q'):
        #break
    #if not ret:
        #print("stream ended")
        #break
    

#stream.release()
#cv2.destroyAllWindows()
