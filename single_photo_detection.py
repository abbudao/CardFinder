#!/usr/bin/python
import os
import sys
import glob
import dlib
from skimage import io
import cv2
import numpy as np
from detect_card import detect_card

def multiple_crop(photopath):
    # Now let's use the detector as you would in a normal application.  First we
    # will load it from disk.
    detector = dlib.simple_object_detector("detector.svm")
    img = io.imread(photopath)
    dets = detector(img)
    detected_rectangles= []
    for k, d in enumerate(dets):
        detected_rectangles.append([d.left(),d.top(),d.right(),d.bottom()])
#        print(detected_rectangles)
    cropped_vector=crop_cards(img, detected_rectangles)
#    print(cropped_vector)
    return cropped_vector


def crop_cards(img, rectanglelist):
    cropped_img=[]
    for i in rectanglelist:
        single_crop=img[i[0]:i[2], i[1]:i[3]]
        cropped_img.append(single_crop) 
    return cropped_img


crops=multiple_crop("Training/7.jpg")
gray=cv2.cvtColor(crops[0],0)
base= cv2.imread("base.jpg",0)
detect_card(gray,base)
