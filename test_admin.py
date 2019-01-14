import numpy as np
import cv2
import math
import pyautogui
#from directkeys import PressKey, ReleaseKey, W, A, S, D, Space, F

capture = cv2.VideoCapture(1)
hand_cascade = cv2.CascadeClassifier('hand.xml')

while capture.isOpened():
    
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (800, 600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Get hand data from the rectangle sub window
    cv2.rectangle(frame,(500,100),(700,300),(0,255,0),0)
    cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)
    crop_image = frame[100:300, 100:300]

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(crop_image, (3,3), 0)
    
    # Change color-space from BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    upper = np.array([20,255,255])
    lower = np.array([2,0,0])
    
    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, lower, upper)
    
    # Kernel for morphological transformation    
    kernel = np.ones((5,5))
    
    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations = 1)
    erosion = cv2.erode(dilation, kernel, iterations = 1)    
       
    # Apply Gaussian Blur and Threshold
    filtered = cv2.GaussianBlur(erosion, (3,3), 0)
    ret,thresh = cv2.threshold(filtered, 127, 255, 0)
    
    # Show threshold image
    #cv2.imshow("Threshold", thresh)

    # Find contours
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    drawing = np.zeros(crop_image.shape,np.uint8)
    
    try:
        # Firing
        hand = hand_cascade.detectMultiScale(gray, 1.3, 5)
        for(x, y, w, h) in hand:
            if x > 500 and y > 100 and x+w < 700 and y+h < 300:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                pyautogui.mouseDown(button='right')
                pyautogui.mouseDown()
            else:
                pyautogui.mouseUp()
                pyautogui.mouseUp(button='right')
    
        # Find contour with maximum area
        contour = max(contours, key = lambda x: cv2.contourArea(x))
        
        # Create bounding rectangle around the contour
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image,(x,y),(x+w,y+h),(0,0,255),0)
        
        # Find convex hull
        hull = cv2.convexHull(contour)
        
        # Draw contour
        cv2.drawContours(drawing,[contour],-1,(0,255,0),0)
        cv2.drawContours(drawing,[hull],-1,(0,0,255),0)
        
        # Find convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour,hull)
        
        # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger 
        # tips) for all defects
        count_defects = 0
        
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14
            
            # if angle > 90 ignore draw a circle at the far point
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image,far,1,[0,0,255],-1)

            cv2.line(crop_image,start,end,[0,255,0],2)

        def release_key(key):
            if key == w:
                  pyautogui.keyUp('a')
                  pyautogui.keyUp('d')
                  pyautogui.keyUp('s')
                  pyautogui.keyUp('space')
                  pyautogui.keyUp('f')
            elif key == a:
                  pyautogui.keyUp('w')
                  pyautogui.keyUp('d')
                  pyautogui.keyUp('s')
                  pyautogui.keyUp('space')
                  pyautogui.keyUp('f')
            elif key == d:
                  pyautogui.keyUp('w')
                  pyautogui.keyUp('a')
                  pyautogui.keyUp('s')
                  pyautogui.keyUp('space')
                  pyautogui.keyUp('f')
            elif key == s:
                  pyautogui.keyUp('w')
                  pyautogui.keyUp('d')
                  pyautogui.keyUp('a')
                  pyautogui.keyUp('space')
                  pyautogui.keyUp('f')
            elif key == space:
                  pyautogui.keyUp('w')
                  pyautogui.keyUp('a')
                  pyautogui.keyUp('d')
                  pyautogui.keyUp('s')
                  pyautogui.keyUp('f')

        # Print keys
        # movement
        if count_defects == 0:
            cv2.putText(frame,"W", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            pyautogui.keyDown('w')
            release_key(w)
        elif count_defects == 1:
            cv2.putText(frame,"A", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            pyautogui.keyDown('a')
            releasse_key(a)
        elif count_defects == 2:
            cv2.putText(frame, "D", (5,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            pyautogui.keyDown('d')
            release_key(d)
        elif count_defects == 3:
            cv2.putText(frame,"S", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            pyautogui.keyDown('s')
            release_key(s)
        elif count_defects == 4:
            cv2.putText(frame,"SPACE/F", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            pyautogui.press('space')
            pyautogui.press('f')
            release_key(space)
            
        else:
            release_key(space)

    except:
        pass

    
    cv2.imshow("Gesture", frame)
    #all_image = np.hstack((drawing, crop_image))
    #cv2.imshow('Contours', all_image)
      
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
