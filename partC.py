import cv2
import dlib
import numpy as np

def faceDectection(frame, face_detector)->np.ndarray:
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(frame_gray,0)
    corners_raw = 0
    if len(faces) > 0:
        for face in faces:
            x, y = face.left(), face.top()
            w, h = face.right() - x, face.bottom() - y
            roi = frame_gray[y:y+h, x:x+w] #Returns cropped region of interest for face
            corners = cv2.goodFeaturesToTrack(roi, maxCorners=50, qualityLevel=0.01, minDistance=10)
            corners_raw = corners
            corners = np.int16(corners)
            # Adjust corner coordinates to the original image
            for corner in corners:
                cx, cy = corner.ravel()
                cv2.circle(frame, (cx + x, cy + y), 5, (0, 255, 0), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            for i in range(len(corners_raw)):
                cx, cy = corners[i].ravel()
                corners_raw[i] = (cx + x, cy + y)
        #cv2.imshow('Main Display',frame)
        return corners_raw
    else:
        return np.array([])
    #end if
face_detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture('clip3.mp4')
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = (0,0,255) #Sets color to red
while (1):
    ret,frame = cap.read()
    h,w,_ = frame.shape
    p0 = faceDectection(frame, face_detector)
    if len(p0)>0:
        #print (p0)
        tracking= True
        baseLine_points = len(p0)
        old_frame = frame
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        it = 1
        while tracking:
            mask = np.zeros_like(old_frame)
            ret ,frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,p0,None,**lk_params)
            print (sum(err))
            print(len(st))
            good_new = p1[st==1]
            good_old = p0[st==1]
            if (baseLine_points - len(good_new)) >= (baseLine_points*0.1) or (sum(err) >500 and it > 1) :
                print("Tracking Lost")
                tracking = False
                break
            # for i,(new,old) in enumerate(zip(good_new,good_old)):
            #     a, b = new.ravel()
            #     c, d = old.ravel()
            #     a = int(a)
            #     b = int(b)
            #     c = int(c)
            #     d = int(d)
            #     mask = cv2.line(mask, (a, b), (c, d), color, 2)
            #     frame = cv2.circle(frame, (a, b), 5, color, -1)
            img = cv2.add(frame, mask)
            min_x, min_y = np.min(good_new, axis=0)
            max_x, max_y = np.max(good_new, axis=0)
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)
            print(min_x,min_y,max_x,max_y)
            img = cv2.rectangle(frame, (min_x, min_y), (max_x,max_y), color, 2)
            img = cv2.resize(img, (w,h))
            cv2.imshow('Main Display',img)
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
            k = cv2.waitKey(33) & 0xff
            it += 1
    else:
        cv2.imshow('Main Display', frame)
        k = cv2.waitKey(33) & 0xff