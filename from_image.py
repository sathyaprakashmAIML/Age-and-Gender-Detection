import cv2
import numpy as np
import imutils

age_weight = r'D:\ML\cv2\deep\age_Detection\age_deploy.prototxt'
age_config = r'D:\ML\cv2\deep\age_Detection\age_net.caffemodel'
gen_weight=r'D:\ML\cv2\deep\age_Detection\gender_deploy.prototxt'
gen_config=r'D:\ML\cv2\deep\age_Detection\gender_net.caffemodel'
gen_list=['Male','Female']

age_net=cv2.dnn.readNet(age_config,age_weight)
age_list=['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
model_mean=(78.4263377603, 87.7689143744, 114.895847746)
gen_net=cv2.dnn.readNet(gen_config,gen_weight)

cascade='haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(cascade)
offset=5
cam=cv2.imread('sathya.jpg')
resize=imutils.resize(cam,width=500,height=500)
gray=cv2.cvtColor(cam,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,1.3,6)
if len(faces)==0:
        cv2.putText(cam,'No face detected',(10,30),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),1)
else:
     for x,y,w,h in faces:
         cv2.rectangle(cam,(x,y),(x+w,y+h),(0,255,0),2)
         img=cam[y:y+h,x:x+w]
         blob=cv2.dnn.blobFromImage(img,1.0,(227,227),model_mean,swapRB=False)
         age_net.setInput(blob)
         age_pred=age_net.forward()
         age=age_list[age_pred[0].argmax()]
         gen_net.setInput(blob)
         gen_pred=gen_net.forward()
         gen=gen_list[gen_pred[0].argmax()]
         print(age)
         print(gen)
         cv2.putText(cam,f'{age},{gen}',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
cv2.imshow('Detection',cam)
cv2.waitKey(0)
cv2.destroyAllWindows()
         
    
