import sys
import cv2
import numpy as np

import matplotlib as mpl
if sys.platform == "darwin":
    mpl.use("Qt5Agg")
import matplotlib.pyplot as plt

from skimage import morphology


def remove_background(img):
    original = img.copy()
    
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
    img = cv2.erode(img, element, iterations = 1)
    img = cv2.dilate(img, element, iterations = 1)
    img = cv2.erode(img, element)

    new_img = np.ones((img.shape[0], img.shape[1]))*255
    new_img[img[:, :, 0]  >= 240] = 0

    r, contours = extract_contours(new_img.astype(np.uint8), min_radius=0, max_radius=1000000, return_contours=True)
    outline = contours[np.argmax(r)]

    mask = np.ones(img.shape[:2], dtype="uint8") 
    cv2.drawContours(mask, [outline], -1, 0, -1)
    mask = cv2.erode(mask, element)

    image = original.copy()
    image[mask != 0] = 255

    # for debugging
    # f, axarr = plt.subplots(ncols=2)
    # axarr[0].imshow(original, cmap="gray")
    # axarr[1].imshow(mask, cmap="gray")
    # plt.show()

    return image


def setup_blob_detector(**kwargs):
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Area.
    params.filterByArea = False
    params.maxArea = kwargs.pop("maxArea", 2000)

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = kwargs.pop("circularity", .1)

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = kwargs.pop("convexity", 0.87)

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    return detector

def extract_contours(img, min_radius=0, max_radius=100, return_contours=False):
    # Extract centroids location from contours
    if int(cv2.__version__[0]) >= 3:
    	_, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
    	contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = [cv2.minEnclosingCircle(cnt) for cnt in contours]
    x = [x for (x,y),r in centroids if r<=max_radius and r >= min_radius]
    y = [y for (x,y),r in centroids if r<=max_radius and r >= min_radius]
    r = [r for (x,y),r in centroids if r<=max_radius and r >= min_radius]

    if not return_contours:
        return img, x, y, r
    else:
        return r, contours

def extract_circles(img, min_radius=0, max_radius=100):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,1,10, param1=50,param2=12,minRadius=min_radius,maxRadius=max_radius)

    if circles is not None:
    # centroids = [cv2.minEnclosingCircle(cnt) for cnt in contours]
        x = [x for x,y,r in circles[0]]
        y = [y for x,y,r in circles[0]]
        r = [r for x,y,r in circles[0]]
    else:
        x, y, r = [], [], []

    return img, x, y, r

def extract_blobs(img, **kwargs):
    detector = setup_blob_detector(**kwargs)
    keypoints = detector.detect(img)

    if not keypoints:
        return img, [], [], []
    else:
        x = [k.pt[0] for k in keypoints]
        y = [k.pt[1] for k in keypoints]
        r = [k.size for k in keypoints]
        return img, x, y, r

def extract_from_image(img, method="blobs", min_radius=0, max_radius=100, **kwargs):
    # Apply closure to the image
    # kernel = np.ones((31,31),np.uint8)
    # closed = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, kernel)

    # Blurr
    img =  cv2.GaussianBlur(img,(11,11),cv2.BORDER_DEFAULT)

    # Apply otsu threshold
    ret, thresh = cv2.threshold(img, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if method == "blobs":
        return extract_blobs(thresh, **kwargs)
    elif method == "circles":
        return extract_circles(thresh)
    elif method == "contours":
        return extract_contours(thresh)
    else:
        raise ValueError("Unknown method {}".format(method))

    