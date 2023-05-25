import cv2
from cvzone.HandTrackingModule import HandDetector
# Starting with Webcam coding
import numpy as np
import math


cap = cv2.VideoCapture(0)  # 0 for id number of webcam
detector = HandDetector(maxHands=1)   # help classify better classify

offset = 20
imgSize= 300
while True:
    success, img = cap.read()
    hands, img= detector.findHands(img)
    if hands:
        hand = hands[0]  # bcoz we only have 1 hand
        # Get bounding box info
        x, y, w, h = hand["bbox"]

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255     # uint8 0 to 255
        imgCrop = img[y-offset:y+h+offset, x-offset: x+w+offset]

        imgCropShape=imgCrop.shape

        aspectRatio = h/w  # If value is above 1 it means height is greater

        if aspectRatio > 1:
            k = imgSize/h    # Stretching the height
            # calculated width according to height
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize  # Height Width and channels are 3 inputs here we give initial and final height and width is required for our problem
        else:
            k = imgSize / w  # Stretching teh height
            # calculated width according to height
            hCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (hCal, imgSize))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)



    cv2.imshow("Image", img)
    cv2.waitKey(1)





