#!/usr/bin/python

import numpy as np
import os
import cv2
from PIL import Image
import pytesseract
import math

def findCorrespondingPoints(generatedArray, template):
	rowG, colG = generatedArray.shape
	rowT, colT = template.shape
	print rowT
	assert (colG == colT and rowT == rowG)
	newGenerated = np.zeros_like(generatedArray)

	for i in range(rowG):
		leastDistance =  None
		bestPoint = None

		for j in range(rowG):
			dist = np.linalg.norm(template[i] - generatedArray[j])
			if leastDistance == None or dist < leastDistance:
				leastDistance = dist
				bestPoint = generatedArray[j]

		newGenerated[i] = bestPoint

	return newGenerated

def topologicalSkeleton(src):
	img = src
	skel = np.zeros_like(img)
	size = np.size(img)

	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

	while True:
		eroded = cv2.erode(img,element)
		temp = cv2.dilate(eroded, element)
		temp = cv2.subtract(img, temp)
		skel = cv2.bitwise_or(skel, temp)
		img = eroded.copy()

		zeros = size - cv2.countNonZero(img)
		if zeros == size:
			return skel

def findLines(src, minLineLength):
	#edges = cv2.Canny(src,10,900,apertureSize = 3)
	height, width = src.shape

	maxLineGap = 5
	lines = cv2.HoughLinesP(src,1,np.pi/180,175,minLineLength,maxLineGap)

	img = src.copy()
	for x in range(0, len(lines)):
    		for x1,y1,x2,y2 in lines[x]:
        			cv2.line(img,(x1,y1),(x2,y2),(0,0,0),2)

	return img
def processLargeSquare(src):
	height, width = src.shape
	src = findLines(src, 10)
	# should be square
	dim = height//3

	sudokuBoard = []
	for w in range(0,height - dim,dim):
		for h in range(0,width - dim,dim):
			processInternalSquare(src[w:w + dim, h:h + dim])

def processInternalSquare(src):
	height, width = src.shape
	# should be square
	dim = height // 3
	innerSquare = []
	print height, width
	for w in range(0, height - dim, dim):
		for h in range(0, width - dim, dim):
			temp =  src[w:w + dim,h: h + dim]
			temp = cv2.bitwise_not(temp)
			#cv2.imshow("smallSquare",temp)
			#cv2.waitKey()
			#classifyNumber(temp)

def classifyNumber(src):
	newImg = Image.fromarray(src)
	txt = pytesseract.image_to_string(newImg, config = " outputbase nobatch digits ")
	print txt

def findBiggestArea(contours):
    biggest = None
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)
        if area > max_area and len(approx) == 4:
            biggest = approx
            max_area = area
    return biggest

if __name__ == "__main__":
	# import the image
	img = cv2.imread('sudoku-original.jpg')
	# perform gaussian blur to smoothen the image
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray,(5,5),0)
        # perform thresholding
	thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
        # find the image contours
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	masking = np.zeros_like(gray)

        # find the biggest area (corresponding to the large sudoku square)
        biggest = findBiggestArea(contours)

        # find the largest area contours on the image mask (to get rid of the content outside of the square)
	cv2.drawContours(masking,[biggest],-1,255,-1)
	thresh = cv2.bitwise_and(thresh,thresh, mask=masking)

        # create a target transformation shape
	rows, cols = thresh.shape
	minDim = min(rows,cols)
	transformed = np.array([[0,minDim], [minDim,0],[minDim,minDim],[0,0]],dtype=np.float32)

        # prepare the biggest controur for transformation
	biggest = np.reshape(np.ravel(biggest), (4,2))
	biggest = biggest.astype(np.float32)

        # make sure that the points match up
	transformed = findCorrespondingPoints(transformed, biggest)

        # warp the image according to the biggest square (should take up the whole image now)
	retValue = cv2.getPerspectiveTransform(biggest, transformed)
	warp = cv2.warpPerspective(thresh, retValue, (minDim,minDim))

        # perform a topological skeleton operation (just the outline)
	skeleton = topologicalSkeleton(warp)

	processLargeSquare(warp)

	warp = cv2.bitwise_not(warp)

	cv2.imshow("original",img)
	cv2.imshow('corrected sudoku' ,warp)
        cv2.imwrite('corrected.png',warp)
	cv2.waitKey()
