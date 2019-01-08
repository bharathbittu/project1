import cv2
import numpy as np

video=cv2.VideoCapture('motion.mp4')
video.set(cv2.CAP_PROP_FRAME_WIDTH,1024 )
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
sc=None
while True:
    ret,frame=video.read()
    if not ret:
     break
    #print(frame)
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    gray_img=cv2.GaussianBlur(gray_img,(21,21),0)
    
    if sc is None:
        sc=gray_img
        continue
    
    diff=cv2.absdiff(sc,gray_img)
    _,diff_thresh=cv2.threshold(diff,5,255,cv2.THRESH_BINARY)
    diff_thresh=cv2.dilate(diff_thresh,None,iterations=2)
    sc=gray_img
    _,cntr,_=cv2.findContours(diff_thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in cntr:
        #print("1")
        if cv2.contourArea(contour) < 5000: 
            continue
        if cv2.contourArea(contour)>100000:
            continue
        
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    out.write(frame)
    cv2.imshow('vid',frame)
out.release()    
