#!/usr/bin/python
import os
import sys
import glob
import dlib
from skimage import io
import cv2
import numpy as np
def detect_single(photopath):
    # Now let's use the detector as you would in a normal application.  First we
    # will load it from disk.
    detector = dlib.simple_object_detector("detector.svm")
    win = dlib.image_window()
    img = io.imread(photopath)
    dets = detector(img)
    detected_rectangles= []
    for k, d in enumerate(dets):
        detected_rectangles.append([d.left(),d.top(),d.right(),d.bottom()])
    print(detected_rectangles)

def crop_card(photopath, rectanglelist) 
    
    detect_single("Training/7.jpg")
