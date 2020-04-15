
import os
import cv2
from WordSegmentation import wordSegmentation, prepareImg
import sys
del sys.argv[0]
print(len(sys.argv))

def main():
	"""reads images from data/ and outputs the word-segmentation to out/"""
	path = sys.argv[0]
	# read input images from 'in' directory
	imgFiles = os.listdir(path)
	for (i,f) in enumerate(imgFiles):
		print('Segmenting words of sample %s'%f)

		# read image, prepare it by resizing it to fixed height and converting it to grayscale
		img = prepareImg(cv2.imread(path+'/%s'%f), 50)

		# execute segmentation with given parameters
		# -kernelSize: size of filter kernel (odd integer)
		# -sigma: standard deviation of Gaussian function used for filter kernel
		# -theta: approximated width/height ratio of words, filter function is distorted by this factor
		# - minArea: ignore word candidates smaller than specified area
		res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)

		# write output to 'out/inputFileName' directory
		#if not os.path.exists('file:///C:/Users/kiruthika.parthiban/Desktop/Data_Standardization/HandwrittenTextRecognition/notes/out/'%f):
			#os.mkdir('file:///C:/Users/kiruthika.parthiban/Desktop/Data_Standardization/HandwrittenTextRecognition/notes/out/'%f)
		b=0
		# iterate over all segmented words
		print('Segmented into %d words'%len(res))
		for (j, w) in enumerate(res):
			(wordBox, wordImg) = w
			(x, y, w, h) = wordBox
			#os.mkdir('C:/Users/kiruthika.parthiban/Documents/lines/word{i}')
			cv2.imwrite(path+'/word{i}/'+str(b)+'.png', wordImg) # save word
			cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
			b+=1

		# output summary image with bounding boxes around words
		cv2.imwrite(path+'/summary.png', img)


if __name__ == '__main__':
	main()