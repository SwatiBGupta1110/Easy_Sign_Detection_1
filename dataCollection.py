import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize the webcam and hand detector
cap = cv2.VideoCapture(0)
hand_detector = HandDetector(maxHands=1)

# Constants for image cropping and resizing
crop_offset = 20
img_size = 200

# Folder to save images and a counter for filenames
output_folder = "Data/"
image_counter = 0

while True:
    try:
        # Read a frame from the camera
        success, frame = cap.read()
        if not success:
            break  # Break the loop if there's no valid frame from the camera

        # Detect hands in the frame
        hands = hand_detector.findHands(frame, draw=False)

        # Create a copy of the frame for output
        frame_output = frame.copy()

        if hands:
            # Get the bounding box of the detected hand
            hand = hands[0]
            x, y, w, h = hand["bbox"]

            # Create a white canvas to place the cropped and resized hand
            white_canvas = np.ones((img_size, img_size, 3), np.uint8) * 255

            # Crop the hand region from the original frame
            hand_region = frame[y - crop_offset:y + h + crop_offset, x - crop_offset:x + w + crop_offset]

            # Determine the aspect ratio of the cropped region
            aspect_ratio = h / w

            if aspect_ratio > 1:
                # If the aspect ratio is greater than 1, crop and resize the hand horizontally
                k = img_size / h
                width_calculated = math.ceil(k * w)
                hand_resized = cv2.resize(hand_region, (width_calculated, img_size))
                width_gap = math.ceil((img_size - width_calculated) / 2)
                white_canvas[:, width_gap:width_calculated + width_gap] = hand_resized
            else:
                # If the aspect ratio is less than or equal to 1, crop and resize the hand vertically
                k = img_size / w
                height_calculated = math.ceil(k * w)
                hand_resized = cv2.resize(hand_region, (img_size, height_calculated))
                height_gap = math.ceil((img_size - height_calculated) / 2)
                white_canvas[height_gap:height_calculated + height_gap, :] = hand_resized

            # Display the cropped hand and the white canvas with the resized hand
            cv2.imshow("HandCrop", hand_region)
            cv2.imshow("HandWhite", white_canvas)

        # Display the original frame with any overlays (drawn hands)
        cv2.imshow("Frame", frame_output)

        # Wait for key press for 1 millisecond
        key = cv2.waitKey(1)

        # Exit the loop if 'q' key is pressed
        if key == ord("q"):
            break

        # Generate the formatted timestamp
        timestamp = time.strftime("%d_%m_%Y_%H_%M_%S")

        # Generate the filename with the formatted timestamp
        image_name = f"{output_folder}/Image_{timestamp}.jpg"

        # Save the image if 's' key is pressed
        if key == ord("s"):
            image_counter += 1
            cv2.imwrite(image_name, white_canvas)
            print(f"Image saved: {image_name}, Counter: {image_counter}")

    except Exception as e:
        # Handle any exception that may occur during processing
        print(f"Error: {e}")

# Release the camera and close all windows when the loop ends
cap.release()
cv2.destroyAllWindows()
