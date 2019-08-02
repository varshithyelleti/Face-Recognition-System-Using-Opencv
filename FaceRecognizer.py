
import numpy as np
import cv2
import time
import sys

start=time.time()
period=8

face_cas = cv2.CascadeClassifier('haarcascade_profileface.xml')
cap = cv2.VideoCapture(0);
recognizer = cv2.face.LBPHFaceRecognizer_create();

recognizer.read('trainer/trainer.yml');
flag = 0;

filename='filename';
dict = {
            'item1': 1
       }
fp=open('Employees_data.txt','r')
Employee_list=fp.readlines()
fp.close()

Employee_list=list(Employee_list)

Employee_dict={}
for i in range(len(Employee_list)):
    Employee_ID,Employee_Name=Employee_list[i].split()
    if(Employee_ID not in Employee_dict):
        Employee_dict[Employee_ID]=Employee_Name
    
Autherized=[]

#font = cv2.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 5, 1, 0, 1, 1)
font = cv2.FONT_HERSHEY_SIMPLEX
Frame_iterations=0
UnAutherizedCount=1

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 7);
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y + h, x:x + w]

        id,conf=recognizer.predict(roi_gray)
        print(conf,end="")
        print(",")
        if(conf>=50):
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255),2);
            cv2.putText(img,"Unknown"+"-"+"UnAuthorized",(x,y-10),font,0.55,(120,120,255),1)
            
            cv2.imwrite("UnAuthorized/Person"+str(UnAutherizedCount)+".jpg", gray[y:y+h,x:x+w])
            UnAutherizedCount+=1
            print
        else:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)
            cv2.putText(img,"SifyEmployee"+"-"+"Authorized",(x,y-10),font,0.55,(120,255,120),1)

        
    cv2.imshow('frame',img);  
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break;

cap.release();
cv2.destroyAllWindows()
