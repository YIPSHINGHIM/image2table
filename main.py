import sys
from io import StringIO

import cv2
import pandas as pd
import pytesseract

# Step 1.1: Read the image
image = cv2.imread('table_image.jpeg')

# Check if the image was properly loaded
if image is None:
    print('Could not open or find the image')
else:
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Step 1.3: Detect the table

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area (this value might need adjusting)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]

    # Assuming the largest contour is the table
    table_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle for the table contour
    x, y, w, h = cv2.boundingRect(table_contour)

    # Step 1.4: Extract the table

    # Extract the table from the image
    table = image[y:y+h, x:x+w]

    # Display the table image
    # cv2.imshow('Table', table)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Convert the table image to string
    table_string = pytesseract.image_to_string(table)

    print(table_string)
