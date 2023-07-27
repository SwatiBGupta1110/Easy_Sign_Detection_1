import cv2
from cvzone.HandTrackingModule import HandDetector # To find hand in image
# Starting with Webcam coding
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)  # 0 for id number of webcam  --> capture object
detector = HandDetector(maxHands=1)   # help classify better classify # Bcoz we want 1 hand for data collection part

offset = 20
imgSize= 200


# what we dont want to do is we dont want to continuoulsy save image, we want to save image when we click a button
#for that we will use S button to save image
#folder that will save our images


folder = "Data/"
counter = 0

while True:
    success, img = cap.read()
    hands= detector.findHands(img, draw= False)           # To find hand in image # You want skeleton as well we will keep skeleton
    # hands, img = detector.findHands(img, draw= False)
    imgOutput = img.copy()
    if hands:                                 # if there is something in hand
        hand = hands[0]  # bcoz we only have 1 hand thats why we write hands [0] if that is the case we get bounding box information from that
        # Get bounding box info
        x, y, w, h = hand["bbox"]  # we have x , y, width, height we are extracting these values from dict

        # We are creating a new image with white background here , it is like a matrix
        # matrix we can create through numpy
        #we are creating matrix of ones
        # we are giving fix size for this 300 by 300 by 3 (bcoz it is a colored image), we are keeping it square
        #we will also give type of values if we dont give that u will not see correct colors
        # in our case bcoz it is a color image our values range from 0 to 255 so it is a 8 bit values
        # so we say numpy of unsigned integer of 8 bits
        # right now our pixel values are 1 bcoz we defined numpy array which is filled with ones
        # we need to multiply by 255 to make it white
        # now image white remains of same size
        # now we need to put cropped img on white img at center
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255     # uint8 0 to 255

        # we are writing dimensions how we want to crop here
        # It is a matrix so we need to define starting height, starting width, ending height, ending width
        # start height is y: so y+ height
        #ending width will be x" so x+width
        #This will give bounding box to us
        # because bounding box was exactly touching tip of our finger, we add some space
        # To add space we leave a offset
        # so adding offset will make image litttle bit bigger than exact values, it will be more appropriate for classifier
        # try keeping hand in between, we can go back and forth
        imgCrop = img[y-offset:y+h+offset, x-offset: x+w+offset]

        # Now as shape of alphabets are different we need to handle this and put all images in one size so that
        # we can pass it to classifier
        imgCropShape=imgCrop.shape # this will give us img crop shape that is height, width and color channels

        # Put img crop matrix inside the img white matrix at these values
        # we already have size of the image so we can use it so staring point will be 0 and height will be height of
        # our cropped image
        # therefore imgWhite[0:imgCrop.shape[0]] = imgCrop
        #imgCrop.shape[0] this will give us whatever the height is
        # same thing we do for width--> start from 0 and then whatever width is is that will be value number 1
        # therefore 0:imgCrop.shape[1]
        #imgWhite[0:imgCrop.shape[0],0:imgCrop.shape[1]] = imgCrop --> This will overlay our cropped image on top of white image
        #but we can see it is not at centerit is so that give similiar resultes for every class
        # So we will check if height bigger than width we will make it 300
        # if width is bigger than height we will strecth width to 300
        # and then we will calculate other value on that basis
        # For example we have vertical height, we will stretch the height to 300 and then to keep the proportion
        # we are going to stretch out the width, and how much we need to stretch out is we need to calculate
        # based on these values i will put it on white image





        aspectRatio = h/w  # If value is above 1 it means height is greater , checking for height

        if aspectRatio > 1: #checking height is greater than 1
            # First thing we need to  do is get the constant, If we stretch out the value to 300
            # what will be the value of width, we need to find that
            #k is constant here
            # imgSize is 300
            # imgSie divided by height , we are stretching height here
            # width calculated will be k multiply by width
            # now this will be a folat but we nee dto round it off, we will tell it to take upper value
            # otherwise it will sometimes take floor, ceil

            k = imgSize/h    # Stretching the height
            # calculated width according to height
            wCal = math.ceil(k*w) # now we have calculated width
            # height is 300
            # imgResize we want to resize the imgcrop and what specific values are
            # the width calculated and height is imgsize which is 300
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))

            # so now if we have vertical image of B the height will be 300 but width will change

            #imgWhite[0:imgResizeShape[0], 0:imgResizeShape[1]] = imgResize
            imgResizeShape = imgResize.shape # image resize shape
            # but we need to put the image at center
            # we have a fix height that is 300 and then we have a different widths
            # and based on that width we need to shift a little bit more
            # so we need to give a little bit gap at very beginning to push it forward at the center
            # we need to find that gap
            # width of our white image is 300 minus is whatever width we calculated that is of the actual image
            # then we divide it by 2 and we add ceiling part so that value is consistent, divide by 2 bcoz we are pushing from first half
            # this is the gap we want to push image to the forward at center

            wGap = math.ceil((imgSize-wCal)/2)
            # height is 300 do nothing keep it like that therefore add colons
            # width will be our gap but ending position is size of the image taht we calculted + plus the width gap
            imgWhite[:, wGap:wCal+wGap] = imgResize  # Height Width and channels are 3 inputs here we give initial and final height and width is required for our problem
            # so now it is the at the center for vertcial images


        else: # adding code for the width in the same manner , we will fix width
            k = imgSize / w  # Stretching the width
            # calculated height according to width
            hCal = math.ceil(k * w)
            # here width is fixed which is imgsize and therefore 300 which is imgSize
            #but height will be calculates
            imgResize = cv2.resize(imgCrop, (imgSize,hCal))
            imgResizeShape = imgResize.shape
            # height gap we give in the same manner
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            # imgWhite[hGap:hCal + hGap, :, :] = imgResize_resized
            # dont go to closee to the image otherwise it wong be able to handle it
            # we can add try except here
        cv2.imshow("ImageCrop", imgCrop) # This will give us another image that we will add as cropped img in a new window
        cv2.imshow("ImageWhite", imgWhite) # This will show us image white



    cv2.imshow("Image", imgOutput)     # To show image
    key = cv2.waitKey(1)              # For 1 millisecond delay
    if key==ord("s"): # If we press s key we will save
        counter +=1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg",imgWhite)
        print(counter) # Just to see how many images it is saving

###WORKING CODE




