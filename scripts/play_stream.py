source = 'rtsp://admin:L2CD8412@192.168.0.103:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif'

import cv2
cap = cv2.VideoCapture(source) 
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
while cap.isOpened(): 
    _, image = cap.read()
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       cap.release()
       cv2.destroyAllWindows()