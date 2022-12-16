import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt ")
offset = 20
imgSize = 300


folder = "Data/good"
counter = 0

labels = ["hello doctor Moaty", "good", "i love you doctor Donkol", "yes"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCorp = img[y-offset:y + h+offset, x - offset:x + w+offset]

        imgCorpShape = imgCorp.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCorp, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[0:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCorp, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset), (150, 50), (255, 0, 255), 4)
        cv2.putText(imgOutput, labels[index], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 2500, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)

        cv2.imshow("imgCorp", imgCorp)
        cv2.imshow("imgWhite", imgWhite)

        if index == 0:
            text = pyttsx3.init()
            text.say("hello doctor Moaty")
            text.runAndWait()

        if index == 1:
            text = pyttsx3.init()
            text.say("good ")
            text.runAndWait()

        if index == 2:
            text = pyttsx3.init()
            text.say("i love you doctor Donkol ")
            text.runAndWait()

        if index == 3:
            text = pyttsx3.init()
            text.say("yes")
            text.runAndWait()

    cv2.imshow("image", imgOutput)
    cv2.waitKey(1)
