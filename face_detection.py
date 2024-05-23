import cv2
import numpy as np
def Detect_Face():
    cap=cv2.VideoCapture(0)
    face_Cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    eye_Cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
    while True:
        ret, frame=cap.read()
        if not ret:
            break
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=face_Cascade.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
            roi_gray=gray[y:y+h,x:x+w]
            roi_color=frame[y:y+h,x:x+w]
            eyes=eye_Cascade.detectMultiScale(roi_gray, 1.3,5)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey),(ex+ew, ey+eh),(0,255,0),5)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1)==ord('q'):
            break
    video=cap.release()
    return video
Detect_Face()