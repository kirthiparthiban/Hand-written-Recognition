import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def char_seg(path,filename):
    imgFiles = os.listdir(path)
    for (i,a) in enumerate(imgFiles):
        img = cv2.imread(path+f"/{a}")
        height, width = img.shape[:2]
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        print("Applying Adaptive Threshold with kernel :- 21 X 21")
        bin_img = cv2.adaptiveThreshold(grey_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 20)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

        print("Noise Removal From Image.........")
        final_thr = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
        # find connected components. OpenCV: return type differs between OpenCV2 and 3
        (components, _) = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # append components to result
        res = []
        for c in components:
          #skip small word candidates
          if cv2.contourArea(c) < 50:
             continue
          #apend bounding box and image of word to result list
          currBox = cv2.boundingRect(c) # returns (x, y, w, h)
          (x, y, w, h) = currBox
          currImg = img[y:y+h, x:x+w]
          res.append((currBox, currImg))

        res = sorted(res, key=lambda entry:entry[0][0])
        print('Segmented into %d words'%len(res))
        b=0
        for (j, w) in enumerate(res):
            (wordBox, wordImg) = w
            (x, y, w, h) = wordBox
            os.makedir(path+f'/words{j}/')
            cv2.imwrite(path+f'/words{j}/'+str(b)+'.png', wordImg) # save word
            cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
            b += 1

        # output summary image with bounding boxes around words
        cv2.imwrite(path+'/character_segmented.png', img)
