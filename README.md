# CPET347_Project_2
Repository for CPET347 Project 2 –Real-Time Face Detection and Tracking  w Classical Computer Vision Techniques 

Required Libraries: OpenCV-Python, numpy, dlib
Each part can function by itself and does not require any other dependecies other than the above libraries.

Project_2: Finished pipelined user selectable video processing tool. The keymappings are displayed on the video output along with the current mode and relavent warnings/ information.
For Project_2(part d), it defaults to the second webcame device as that was it was developed for. change "cap = cv2.VideoCapture(1)" -> "cap = cv2.VideoCapture(0)" if you only have one webcam device, otherwise set the parameter to reference the device you would like to use.

Part A: Simple foreground/background seperation taking a video input and outputting a background referecne and a foreground video outpt. Change the cap.cv2.VideoCaptue(PATH) such that path references the video you would like to process.

Part B: Viola-Jones(Haar Cascades) and Histogram of Oriented Gradients(HOG) based face detectors. Change cap = cv2.VideoCapture(PATH) such that PATH references the video you would like to process. To switch between the two algorithims, comment/uncomment the function calls in the main method

Part C: Optical Flow Face Tracking. Change cap = cv2.VideoCapture(PATH) such that PATH references the video you would like to process. This will first find a face using the HOG face detector, once it has detected one, it will track the face until it loose track or their is a significat enough error. This will then re-detect a face repeating the process.

For each part a, b ,c the video refereces for "cap" will need to be changed to reference whatever video you are going to process.


