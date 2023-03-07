import cv2
import dlib
import numpy as np
import sklearn.model_selection
from keras.models import load_model
model=load_model('deepfakeobama.h5')

counter=0
ones=0
zeroes=0
camera=cv2.VideoCapture("obamaoriginal3.mp4")
camera.set(3,600)
camera.set(4,400)
detector=dlib.get_frontal_face_detector()
landmarkpred=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while camera.isOpened():
    status,frame=(camera.read())
    alldatalist=[]
    if status==True:
        smallframe=cv2.resize(frame,(600,400))
        greyimage=cv2.cvtColor(smallframe,cv2.COLOR_BGR2GRAY)
        faces=detector(greyimage)
        for n in faces:
            counter=counter+1
            landmarks=landmarkpred(greyimage,n)
            points=[]
            for facialpoint in range(0,68,1):
                xcoor=landmarks.part(facialpoint).x
                ycoor=landmarks.part(facialpoint).y
                points.append([xcoor,ycoor])
                alldatalist.append(xcoor)
                alldatalist.append(ycoor)
            xdistance=points[16][0]-points[0][0]
            ydistance=points[8][1]-points[19][1]
            ratio=xdistance/ydistance
            alldatalist.append(ratio)
            alldatalist=np.array(alldatalist)
            alldatalist=np.reshape(alldatalist,(1,137))
            prediction=model.predict(alldatalist)
            print(np.argmax(prediction))
            if np.argmax(prediction)==0:
                zeroes=zeroes+1
            elif np.argmax(prediction)==1:
                ones=ones+1
            cv2.putText(smallframe,str(np.argmax(prediction)), (300,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.imshow("frame",smallframe)
    if cv2.waitKey(30) & 0xFF==ord("q"):
        break

print(counter)
if counter==0:
    print("deepfake")
print(ones,zeroes)

camera.release()
cv2.destroyAllWindows()


#x0,y0,x1,y1,x2,y2...x67,y67,ratio
