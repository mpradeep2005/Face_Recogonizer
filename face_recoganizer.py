import cv2 as cv
import os
import numpy as np

people=[]
for person in os.listdir(r"E:/Face_Recogonizer/res"):
    people.append(person)

harr_cascade=cv.CascadeClassifier("E:/Face_Recogonizer/haar.xml")

face_rec=cv.face.LBPHFaceRecognizer_create()
face_rec.read("E:/Face_Recogonizer/face_trained.yml")

img=cv.imread("E:/Face_Recogonizer/unknown.jpg")

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

face_point=harr_cascade.detectMultiScale(gray,1.1,1)


for (x,y,w,h) in face_point:
    face_roi=gray[y:y+h,x:x+w]
    label,confidence=face_rec.predict(face_roi)
    print(confidence)
    cv.putText(img,str(people[label]),(x//2,y//2),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,0,255),2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,200,0),6)

cv.imshow("finded",img)
cv.waitKey(0)