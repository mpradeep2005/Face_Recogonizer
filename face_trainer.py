import os
import cv2 as cv
import numpy as np

people=[]
for i in os.listdir(r"E:/Face_Recogonizer/res"):
    people.append(i)

dir="E:/Face_Recogonizer/res"
features=[]
labels=[]
harr_cascade=cv.CascadeClassifier("E:/Face_Recogonizer/haar.xml")
for person in people:
    path=os.path.join(dir,person)
    label=people.index(person)

    for imgs in os.listdir(path):
        img_path=os.path.join(path,imgs)

        img=cv.imread(img_path)
        
        if img is None:
            continue

        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        face_roi=harr_cascade.detectMultiScale(gray,1.1,3)

        for (x,y,w,h) in face_roi:
            face=gray[y:y+h,x:x+w]
            features.append(face)
            labels.append(label)

features=np.array(features,dtype="object")
labels=np.array(labels)

trainer=cv.face.LBPHFaceRecognizer_create()
trainer.train(features,labels)

trainer.save("face_trained.yml")
np.save("features.npy",features)
np.save("labels.npy",labels)