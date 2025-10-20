#This is the fully implemented pipline described in the project brief as "Part D"
#Developed in symphonic fashion with help from caffine, nicotine, and some Billy Joel
import cv2
import dlib
import numpy as np

def HOGdetectAndDisplay(frame, face_detector) -> np.ndarray: #Simple implementation of HOG face detection
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts frame to gray scale
    faces = face_detector(frame_gray,1) #Runs face detection
    for face in faces: #Draws rectangles on each found face
        x, y = face.left(), face.top()
        w, h = face.right() - x, face.bottom() - y
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    #Next Face
    return frame
#End Sub
def faceDectection(frame, face_detector)->np.ndarray: #Specialized function to give the point readouts for Tracking
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converts frame to gray scale
    faces = face_detector(frame_gray,0) #Runs face detector
    corners_raw = 0
    if len(faces) > 0: #If there was a face found
        for face in faces:
            x, y = face.left(), face.top() #Gets coordinates for Face
            w, h = face.right() - x, face.bottom() - y
            roi = frame_gray[y:y+h, x:x+w] #Returns cropped region of interest for face
            corners = cv2.goodFeaturesToTrack(roi, maxCorners=50, qualityLevel=0.01, minDistance=10) #runs the feature detection on the ROI
            corners_raw = corners
            corners = np.int16(corners)
            # Adjust corner coordinates to the original image
            for corner in corners: #Normalizes the coordinates to the whole image
                cx, cy = corner.ravel()
                cv2.circle(frame, (cx + x, cy + y), 5, (0, 255, 0), -1)
            #Next Corner
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) #Draws Recatangle for debugging
            for i in range(len(corners_raw)): #Normalizes the coordinates for output
                cx, cy = corners[i].ravel()
                corners_raw[i] = (cx + x, cy + y)
            #Next I
        #Next Face
        return corners_raw
    else:
        return np.array([])
    #end if
# 'B' for background/foreground mask view
# 'd' for face detecion mode
# 't" for tracking mode
# 'w' for straight feed
# 'q' to exit
#basic flow idea : state machine esc program which uses the four above states to generate a Frame that is only shown in one location at any time, only one outer while loop to control process flow and interrupts




face_detector = dlib.get_frontal_face_detector() #Loads the HOG trained model detector
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) #Configures the parameters for the optical flow model
tackClr = (0,0,255) #Sets the track color to red
detectClr = (0,255,0) #Sets the detection color to green
currState = "NONE" #Sets the deafult state to SF
p0 = [] #Init. point array 0
gmm = cv2.createBackgroundSubtractorMOG2(1000,16,True)  # Creates background GMM model
tracking = False #Init. tracking to false
cap = cv2.VideoCapture(1) #Sets up the video capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ret,frame = cap.read() #Gets a sample frame to determine window size
h,w,_ = frame.shape #Gets window size
while(1): #Main while loop which holds the state machine
    ret, frame = cap.read() #Reads in frame from video capture]\
    gmm.apply(frame)  #Adds frame to the gaussian model
    match currState: #Sustaning Machine, state setup does not occur here
        case "NONE": #Straight frane passthrough
            outFrame = frame
        #End Case
        case "BACKGROUND":  #Background / Foreground Separation
            background_model = gmm.getBackgroundImage() #Gets the current background
            gray_bg = cv2.cvtColor(background_model, cv2.COLOR_BGR2GRAY) #Converts background to gray
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converts frame to gray
            diff = cv2.absdiff(gray_bg, gray_frame) #Gets the difference between background and frame
            _, mask = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY) #Threshold to get the binary mask
            mask = cv2.dilate(mask, None, iterations=2)
            foreground = cv2.merge([mask, mask, mask]) #Merges the masks to form the foreground image
            outFrame = foreground #Displays the foreground mask
        #End Case
        case "FACE DETECTION": #Face Detection
            outFrame = HOGdetectAndDisplay(frame, face_detector) #Runs face detection and returns boxed frame
        #End Case
        case "FACE TRACKING": #Tracking Mode
            if tracking: #If tracking is active
                mask = np.zeros_like(old_frame) #Gets the mask from the old frame
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converts the frame to gray
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params) #Runs the optical flow function
                #print(sum(err))
                #print(len(st))
                good_new = p1[st == 1] #Sets the poins for the new and old point arrays
                good_old = p0[st == 1]
                if (baseLine_points - len(good_new)) >= (baseLine_points * 0.1) or (sum(err) > 500 and it > 1): #Loss of tracking function, either a 10% loss in tracked points or an sum error > 500
                    print("Tracking Lost")
                    tracking = False
                #End if
                img = cv2.add(frame, mask) #Adds the mask to the frame
                min_x, min_y = np.min(good_new, axis=0) #Gets the bounding box corrdiantes for the point arrays
                max_x, max_y = np.max(good_new, axis=0)
                min_x, min_y = int(min_x), int(min_y)
                max_x, max_y = int(max_x), int(max_y)
                img = cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), tackClr, 2)  #Draws a rectangle around the tracked target
                img = cv2.resize(img, (w, h)) #Resizes the masked image to the frame size
                old_gray = frame_gray.copy() #Saves the current frame for the next iteration
                p0 = good_new.reshape(-1, 1, 2) #Saves the current points for the next iteration
                it += 1
                outFrame = img #Outputs the image with the tracking box
            else: #If tracking is false
                p0 = faceDectection(frame, face_detector) #Runs the HOG face detector
                if len(p0) > 0: #If the array is not empty
                    tracking = True #Enables tracking
                    baseLine_points = len(p0) #sets the baseline points
                    old_frame = frame #Saves the current frame for the tracking process
                    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY) # Saves the current gray frame for the tracking process
                    it = 1 #Sets tracking iterations to 1
                else: # If there is no face detected
                    trackText = "TRACKING LOST: REACQUIRING FACE"
                    outFrame = frame
                    cv2.putText(outFrame,trackText,(250,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2) #Puts text on screen to display to user that that tracking is disabled
                #end if
            #End if
        #End Case
    #End Select
    text = f"Current Mode: {currState}  | Input Options: S(Video Feed) | B(Background/Foreground) | D(Face Detection) | T(Face Tracking) | Q(Quit Program) | " #Menu Text
    cv2.putText(outFrame, text, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) #Puts menu text on output frame
    cv2.imshow('Project 2 Main Display', outFrame) #Displays output frame
    k = cv2.waitKey(30) & 0xff #Gets keystoroke
    match k: #Next State Logic
        case 98: #B
            nextState = "BACKGROUND"
        #End Case
        case 100: #D
            nextState = "FACE DETECTION"
        #End Case
        case 116: #T
            nextState = "FACE TRACKING"
        #End Case
        case 113: #Q
            nextState = "QUIT"
        #End Case
        case 115: #S
            nextState = "NONE"
        #End Case
        case _: #If no mapped key is pressed cycles state
            nextState = currState
        #End Case
    #End Select
    if nextState != currState: #If the next state is not the current state
        print(nextState) #Debug
        match nextState: #Next State actions
            case "FACE TRACKING":
                p0 = [] #Resets p0 point array
                tracking = False #Defaults to tracking disabled to acquire new face
            #End Case
            case "QUIT":
                cap.release()
                cv2.destroyAllWindows()
                break
            #End Case
        #End Select
    #End if
    currState = nextState #Syncs current state
#End While