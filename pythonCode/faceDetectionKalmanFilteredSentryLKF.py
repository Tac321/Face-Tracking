# tac
# 12 / 8 / 2022 
import numpy as np
import cv2
import math
import time
import serial 
import imtools

# https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml

# pitch and yaw command
arduinoData = serial.Serial('com3', 9600)

#initialize the servo angles
uEst = 320
vEst = 240

# initialize errors
yawProportionalError =0
pastYawProportionalError =0

# Opening serial transmission capability for commanding Arduino servos, using serial characters
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)

uMeasm=0
vMeasm=0

#def initialize(self,YH=0,YDH=0,XNTH=0,YHy=0
fKalman = imtools.LKalman()
fKalman.initialize(uEst,0,0,vEst)
centerX = 320
centerY = 240
#initialized real-time Delta time prior to calculating it and using in the common filter for real 
dt1=0.2
    
while 1:
    
    time_kp1 = time.time()
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    
    #detect amount of faces:
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        # cv2.circle(image, center, radius, color, thickness, lineType,shift)
        uMeasm= x+w//2
        vMeasm= y+h//2
        cv2.circle(img, (uMeasm,vMeasm), 10, [33,255,120], -1) # Pick a color 'green'
        
        
    faceDetected = len(faces)>0
    #print(' faceDetected: \n', faceDetected)
    #print(' uMeas: \n', uMeas)
    
    
    
    # draw Kalman estimate on image. regardless of whether face is detected or not
    #if target is invisible for X seconds, freeze LKF estimate 
    #def runKalman(self,uMeas, vMeas, measAvailable, deltaTime=0.03, noiseX=400, noiseY=400, accelMaxTargEst=1000):
    panNoise = 100
    accNoise = 1000 # Lower this to decrease speed towards measurement
    uEst, vEst = fKalman.runKalman(uMeasm, vMeasm, faceDetected, dt1,panNoise,panNoise,accNoise)
    uEst = int(uEst)
    vEst = int(vEst)
    
    # begin tracking section
    #''' 

    # convert the above into angles for the servos, requires calibration, coordinate frame transformation/normalization.
    # take the difference between the target and the center of the camera
    # print out  for calibration
    # make int
    # yawDifference   = str(math.floor (uEst*(-45/573)+117.9 ))
    
    if(faceDetected): #hack this
    
        # yaw proportional control
        pastYawProportionalError = yawProportionalError
        # yawProportionalError = uMeasm-centerX
        yawProportionalError = uEst-centerX
        derivativeErrorGain = -1.1*0
        # yawDifference   = str(int(math.floor ((yawProportionalError)*-23/320*.5)) + derivativeErrorGain*(yawProportionalError-pastYawProportionalError))
        yawDifference   = str(int(math.floor ((yawProportionalError)*-23/320*.5)))
        
        # add a deadzone to the proportional error
        if(abs((yawProportionalError)*-23/320*.5)<2):
            yawDifference ='0'
           
        # print('yawDiff', yawDifference) # string variable    
        #print('yawDoubleError', (yawProportionalError)*-23/320*.5) 
    
        # pitch any all positions ready to be sent
        send = yawDifference #character buffer send format
        send = send + '\r'
        # arduinoData.write(send.encode('utf-8'))
        arduinoData.write(send.encode())
        # time.sleep(.001)  # calibrate 
    
    ''' '''
    # end tracking section
    
    # draw lfk estimate on image
    cv2.circle(img, (uEst,vEst), 10, [255,0,255], -1) # color red
    # Draw image center circle.
    cv2.circle(img,(centerX,centerY), 5, [0,255,255], 2)
    #get the real Delta time of this program 
    #time_kp2 = time.time()
    #dt1 = (time_kp2 - time_kp1)
    #print("deltaTime: ", dt1)
    
    cv2.imshow('img',img)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

