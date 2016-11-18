#!/usr/bin/python
import os
import sys
import glob
import dlib
from skimage import io

def detect(testpath="Testing/"):
    test_folder=testpath
    testing_xml_path = os.path.join(test_folder, "testing.xml")
# Now let's use the detector as you would in a normal application.  First we
# will load it from disk.
    detector = dlib.simple_object_detector("detector.svm")

# We can look at the HOG filter we learned.  It should look like a face.  Neat!
    win_det = dlib.image_window()
    win_det.set_image(detector)

# Now let's run the detector over the images in the faces folder and display the
# results.
    print("Showing detections on the images in the testing folder...")
    win = dlib.image_window()
    for f in glob.glob(os.path.join(test_folder, "*.jpg")):
        print("Processing file: {}".format(f))
        img = io.imread(f)
        dets = detector(img)
        print(dets)
        print("Number of cards detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(dets)
        dlib.hit_enter_to_continue()

detect()
