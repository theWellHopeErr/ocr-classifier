##############################
# Usage:
# python ocr.py img.ext
##############################
import sys
import os
import json

import cv2
import pytesseract
from pytesseract import Output
from dotenv import load_dotenv

load_dotenv(verbose=True)

pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSARACT_PATH")
print("[INFO] loading image...")
img = cv2.imread(sys.argv[1])

print("[INFO] processing image...")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh1 = cv2.threshold(
    gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

contours, hierarchy = cv2.findContours(
    dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

im2 = img.copy()

file = open("recognized.json", "w+")
file.write("[")
file.close()

for idx, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)

    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cropped = im2[y:y + h, x:x + w]

    file = open("recognized.json", "a")

    print("[INFO] reading text...")
    text = pytesseract.image_to_data(cropped, output_type=Output.DICT)

    print("[INFO] writing text...")
    file.write(json.dumps(text))
    if idx != len(contours) - 1:
        file.write(",")
    else:
        file.write("]")
    # print(text)

    file.close()
print("[Finished] Data written to the file .\\recognised.json")