import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def line_seg(path,filename):
    # Image sectioning
    image = cv2.imread(path+'/'+filename)
    height, width = image.shape[:2]
    image.shape

    # Let's get the starting pixel coordiantes (top left of cropped bottom)
    start_row, start_col = int(height * .3), int(0)
    # Let's get the ending pixel coordinates (bottom right of cropped bottom)
    end_row, end_col = int(height * .7), int(width)
    cropped_img = image[start_row:end_row , start_col:end_col]
    cv2.imwrite(path+"/sectioned_image.png", cropped_img)

    # Preproces the sectioned image
    src_img = cv2.imread(path+"/sectioned_image.png")
    copy = src_img.copy()
    #(1009, 2559, 3) = (742,1320)

    #(360, 1200, 3)
    height = src_img.shape[0]
    width = src_img.shape[1]

    print("\n Resizing Image........")
    src_img = cv2.resize(copy, dsize =(1320, 742), interpolation = cv2.INTER_AREA)
    #src_img = cv2.resize(copy, dsize =(1320, 742), interpolation = cv2.INTER_AREA)
    #src_img = cv2.resize(copy, dsize =(700, 650), interpolation = cv2.INTER_AREA)
    src_img.shape
    height = src_img.shape[0]
    width = src_img.shape[1]

    print("#---------Image Info:--------#")
    print("\tHeight =",height,"\n\tWidth =",width)
    print("#----------------------------#")
    grey_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    print("Applying Adaptive Threshold with kernel :- 21 X 21")
    bin_img = cv2.adaptiveThreshold(grey_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,20)
    bin_img1 = bin_img.copy()
    bin_img2 = bin_img.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kernel1 = np.array([[1,0,1],[0,1,0],[1,0,1]], dtype = np.uint8)

    print("Noise Removal From Image.........")
    final_thr = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    contr_retrival = final_thr.copy()

    print("Beginning Character Semenation..............")
    count_x = np.zeros(shape= (height))
    for y in range(height):
    	for x in range(width):
    		if bin_img[y][x] == 255 :
    			count_x[y] = count_x[y]+1

    print("Beginning Line Segmenation..............")

    # Find the empty white lines and text lines
    text_lines = []
    empty_lines = []
    for i,x in enumerate(count_x):
        if x > 0:
            text_lines.append(i)
        elif x == 0:
            empty_lines.append(i)

    # Find the start position of the each line
    upperlines = []
    for i, val in enumerate(text_lines):
        if val != text_lines[i-1]+1:
           upperlines.append(val)

    # Find the end position of the each line
    lowerlines = []
    for idx, v in enumerate(empty_lines):
        if v != empty_lines[idx-1]+1:
           lowerlines.append(v)

    # Delete any end position value greater than start position for minute pixel ranges
    if upperlines[0] > lowerlines[0]:
        del lowerlines[0]

    for i, j in zip(upperlines, lowerlines):
        if j-i <= 10:
            upperlines.remove(i)
            lowerlines.remove(j)

    print(" Start and end position of each lines : ", upperlines, lowerlines)
    # print(upperlines, lowerlines)
    if len(upperlines)==len(lowerlines):
    	lines = []
    	for y in upperlines:
    		final_thr[y][:] = 255
    	for y in lowerlines:
    		final_thr[y][:] = 255
    	for y in range(len(upperlines)):
    		lines.append((upperlines[y], lowerlines[y]))

    else:
    	print("Too much noise in image, unable to process.\nPlease try with another image. Ctrl-C to exit:- ")
    	exit()

    lines = np.array(lines)
    no_of_lines = len(lines)
    print("\nGiven Text has   # ",no_of_lines, " #   no. of lines")
    lines_img = []

    for i in range(no_of_lines):
    	lines_img.append(bin_img2[lines[i][0]:lines[i][1], :])

    #-------------Letter Width Calculation--------#
    contours, hierarchy = cv2.findContours(contr_retrival,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    final_contr = np.zeros((final_thr.shape[0],final_thr.shape[1],3), dtype = np.uint8)
    cv2.drawContours(src_img, contours, -1, (0,255,0), 1)

    def letter_width(contours):
    	letter_width_sum = 0
    	count = 0
    	for cnt in contours:
    		if cv2.contourArea(cnt) > 20:
    			x,y,w,h = cv2.boundingRect(cnt)
    			letter_width_sum += w
    			count += 1

    	return letter_width_sum/count

    mean_lttr_width = letter_width(contours)
    print("\nAverage Width of Each Letter:- ", mean_lttr_width)
    # Save the lines
    for idx, val in enumerate(lines_img):
        cv2.imwrite(path+f'/lines{idx}.png',255-lines_img[idx])