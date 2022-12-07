import cv2
import time


camera = 'tcp://0.0.0.0:5000'
stream = cv2.VideoCapture(camera)

#if stream is not opened print no stream 
if not stream.isOpened(): 
    print("no stream")
    exit()

fps = int(stream.get(cv2.CAP_PROP_FPS))
width = int(stream.get(3))
height = int(stream.get(4))
size = (width, height)

#write video (save)
output = cv2.VideoWriter('videoStorageOpencv/stream.avi',cv2.VideoWriter_fourcc('M','J','P','G'),fps,size)



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