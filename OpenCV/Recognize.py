import cv2
import numpy as np
from PIL import Image
import pickle
import sqlite3

faceDetect=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
# cam.set(3, 640)
# cam.set(4, 480)
minW=0.1*cam.get(3)
minH=0.1*cam.get(4)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("trainingData/trainingData.yml")
id=0
#set text style
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (203,23,252)

#get data from sqlite by ID
def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

while(True):
    #camera read
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if (conf<100):
            profile=getProfile(id)
            conf=" {0}%".format(round(100-conf))
        else:
            profile=None
            id="unknown"
            conf=" {0}%".format(round(100-conf))
        #set text to window
        if(profile!=None):
            cv2.putText(img, str(id), (x+5,y-5), fontface, fontscale, fontcolor ,2)
            cv2.putText(img, "Name: " + str(profile[1]), (x,y+h+30), fontface, fontscale, fontcolor ,2)
            cv2.putText(img, str(conf), (x+5,y+h-5), fontface, fontscale, fontcolor ,2)
        else:
            cv2.putText(img, str(id), (x+5,y-5), fontface, fontscale, fontcolor ,2)
            cv2.putText(img, "Name: Unknown", (x,y+h+30), fontface, fontscale, fontcolor ,2)
            cv2.putText(img, str(conf), (x+5,y+h-5), fontface, fontscale, fontcolor ,2)
        
        cv2.imshow('FaceRecog',img) 
    k=cv2.waitKey(10)&0xff
    if (k == 27):
        break
print("\nExit")
cam.release()
cv2.destroyAllWindows()