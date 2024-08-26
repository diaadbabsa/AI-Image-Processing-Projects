import cv2
import numpy as np
import Utlis

# Image and camera settings
useWebCam = True
imagePath = "1.jpg"
camera = cv2.VideoCapture(0)
camera.set(10, 160)
imageHeight = 640
imageWidth = 480

# Initialize trackbars and counter values
Utlis.initializeTrackbars()
imageCounter = 0

while True:

    # Capture image from webcam or load it from file
    if useWebCam:
        success, frame = camera.read()
    else:
        frame = cv2.imread(imagePath)
    
    # Resize the image
    frame = cv2.resize(frame, (imageWidth, imageHeight))
    
    # Create a blank image for testing
    blankImage = np.zeros((imageHeight, imageWidth, 3), np.uint8)
    
    # Convert image to grayscale
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blurredFrame = cv2.GaussianBlur(grayFrame, (5, 5), 1)
    
    # Get trackbar values for thresholding
    thresholds = Utlis.valTrackbars()
    
    # Apply Canny Edge Detection
    thresholdFrame = cv2.Canny(blurredFrame, thresholds[0], thresholds[1])
    
    # Apply dilation and erosion
    kernel = np.ones((5, 5))
    dilatedFrame = cv2.dilate(thresholdFrame, kernel, iterations=2)
    thresholdFrame = cv2.erode(dilatedFrame, kernel, iterations=1)

    # Find all contours in the image
    contoursFrame = frame.copy()
    largestContourFrame = frame.copy()
    contours, hierarchy = cv2.findContours(thresholdFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contoursFrame, contours, -1, (0, 255, 0), 10)

    # Find the largest contour
    largestContour, maxArea = Utlis.biggestContour(contours)
    if largestContour.size != 0:
        largestContour = Utlis.reorder(largestContour)
        cv2.drawContours(largestContourFrame, largestContour, -1, (0, 255, 0), 20)
        largestContourFrame = Utlis.drawRectangle(largestContourFrame, largestContour, 2)
        
        pts1 = np.float32(largestContour)
        pts2 = np.float32([[0, 0], [imageWidth, 0], [0, imageHeight], [imageWidth, imageHeight]])
        
        transformationMatrix = cv2.getPerspectiveTransform(pts1, pts2)
        warpedFrame = cv2.warpPerspective(frame, transformationMatrix, (imageWidth, imageHeight))

        # Remove 20 pixels from each side
        warpedFrame = warpedFrame[20:warpedFrame.shape[0] - 20, 20:warpedFrame.shape[1] - 20]
        warpedFrame = cv2.resize(warpedFrame, (imageWidth, imageHeight))

        # Apply adaptive thresholding
        warpedGrayFrame = cv2.cvtColor(warpedFrame, cv2.COLOR_BGR2GRAY)
        adaptiveThresholdFrame = cv2.adaptiveThreshold(warpedGrayFrame, 255, 1, 1, 7, 2)
        adaptiveThresholdFrame = cv2.bitwise_not(adaptiveThresholdFrame)
        adaptiveThresholdFrame = cv2.medianBlur(adaptiveThresholdFrame, 3)

        # Organize images for display
        displayArray = ([frame, grayFrame, thresholdFrame, contoursFrame],
                        [largestContourFrame, warpedFrame, warpedGrayFrame, adaptiveThresholdFrame])

    else:
        displayArray = ([frame, grayFrame, thresholdFrame, contoursFrame],
                        [blankImage, blankImage, blankImage, blankImage])

    # Set labels for display
    labels = [["Original", "Gray", "Threshold", "Contours"],
              ["Largest Contour", "Warp Perspective", "Warp Gray", "Adaptive Threshold"]]

    # Stack images for display in one window
    stackedImages = Utlis.stackImages(displayArray, 0.75, labels)
    cv2.imshow("Result", stackedImages)

    # Save image when 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage" + str(imageCounter) + ".jpg", warpedFrame)
        cv2.rectangle(stackedImages, ((int(stackedImages.shape[1] / 2) - 230), int(stackedImages.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImages, "Scan Saved", (int(stackedImages.shape[1] / 2) - 200, int(stackedImages.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImages)
        cv2.waitKey(300)
        imageCounter += 1
