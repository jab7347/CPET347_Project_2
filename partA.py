import cv2
import numpy as np

def detect_background_gmm(video_path: str, history: int = 500, var_threshold: int = 16,
                          detect_shadows: bool = True) -> np.ndarray:
    print("\n--- Starting GMM Background Detection ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    # Create a GMM background subtractor from OpenCV
    subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold,
                                                    detectShadows=detect_shadows)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process each frame to train the GMM
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # The 'apply' method updates the background model with the current frame
        subtractor.apply(frame)
        print(f"Training GMM with frame {i + 1}/{total_frames}", end='\r')

    # After processing frames, retrieve the learned background model
    background_model = subtractor.getBackgroundImage()

    print("\nGMM background detection complete.")
    cap.release()
    return background_model
def extract_foreground(video_path: str, background_model: np.ndarray, threshold_value: int, output_path: str):
    """
    Extracts the foreground from a video given a background model.


    Args:
        video_path (str): Path to the input video.
        background_model (np.ndarray): The pre-computed background image.
        threshold_value (int): Sensitivity for change detection (0-255).
        output_path (str): Path to save the output video.
    """
    print(f"\n--- Extracting Foreground using provided background model ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    # Setup video writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare the background model (grayscale and blur for better diff)
    gray_bg = cv2.cvtColor(background_model, cv2.COLOR_BGR2GRAY)
    gray_bg = cv2.GaussianBlur(gray_bg, (21, 21), 0)

    # Process video frame by frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        # Compute difference, threshold, and create a mask
        diff = cv2.absdiff(gray_bg, gray_frame)
        _, mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=2)
        foreground = cv2.merge([mask, mask, mask])
        # Apply the mask to the original color frame
        #foreground = cv2.bitwise_and(frame, frame, mask=mask)
        out.write(foreground)
        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames} for foreground extraction", end='\r')

    print(f"\nForeground extraction complete. Video saved to '{output_path}'.")
    cap.release()
    out.release()


inputPath = 'partA_Raw.mp4'
imgoutputPath = 'bgimage.jpg'
outputPath = 'fgimage.mp4'
gmmModel = detect_background_gmm(inputPath)
cv2.imwrite(imgoutputPath,gmmModel)
extract_foreground(inputPath, gmmModel,45, outputPath)
cv2.destroyAllWindows()