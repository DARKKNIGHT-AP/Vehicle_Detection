import cv2


cap=cv2.VideoCapture("highway.mp4")


#object detection form stable camera
object_detector=cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret,frame=cap.read()
    height,width, _ = frame.shape
    print(height,width)

    #extract region of interest
    roi = frame[340:720,500: 800]

    #1. object detection
    mask=object_detector.apply(roi)
    _,mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _=cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        #calculate area and remove small elements
        area=cv2.contourArea(cnt)

        if area > 100:
           x,y,w,h=cv2.boundingRect(cnt)
           cv2.rectangle(roi, (x, y),(x+w, y+h),(0,255,0),3)

    cv2.imshow("Frame",frame)
    key=cv2.waitKey(30)
    if key==27:
        break
    
cap.release()
cv2.destroyAllWindows()
