from __future__ import print_function
import cv2 as cv
import dlib
from skimage.feature import hog
from sklearn.svm import LinearSVC

import argparse
def HCdetectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 4)
        #faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        # eyes = eyes_cascade.detectMultiScale(faceROI)
        # for (x2,y2,w2,h2) in eyes:
        #     eye_center = (x + x2 + w2//2, y + y2 + h2//2)
        #     radius = int(round((w2 + h2)*0.25))
        #     frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
    cv.imshow('Capture - HC Face detection', frame)
def HOGdetectAndDisplay(frame, face_detector):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detector(frame_gray,1)
    for face in faces:
        x, y = face.left(), face.top()
        w, h = face.right() - x, face.bottom() - y
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv.imshow('Capture - HOG Face detection', frame)
face_cascade_name = 'C:/Users/thetr/Documents/CPET347/Project_2/.venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml'
# eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
face_detector = dlib.get_frontal_face_detector()
# eyes_cascade = cv.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
# if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
#     print('--(!)Error loading eyes cascade')
#     exit(0)
#-- 2. Read the video stream
#cap = cv.VideoCapture(1)
cap = cv.VideoCapture('go.mp4')
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
ret, frame = cap.read()
cv.imshow('Capture - HC Face detection',frame)
cv.waitKey(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    #HCdetectAndDisplay(frame)
    HOGdetectAndDisplay(frame, face_detector)
    if cv.waitKey(10) == 27:
        break