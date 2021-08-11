from typing import Pattern
import numpy as np
import cv2
from PIL import ImageGrab
import os
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import os

#take screenshot of the game window
bbox=(00,0,1920,1080)
img = ImageGrab.grab()
imgnp = np.array(img)
img = np.array(img)

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

l_b = np.array([0,0,0])
u_b = np.array([0,0,255])

mask = cv2.inRange(hsv, l_b, u_b)
res = cv2.bitwise_and(img, img, mask=mask)

#save the contours
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgray, contours, -1, (0, 255, 0), 3)




BLACK_THRESHOLD = 200
THIN_THRESHOLD = 30
handname=[]
filename = []
idx = 0
player = 0
HANDHEIGHT = [92,93,94,95,96,]
HANDWIDTH= [115,116,117,118,119,130,131,132]
#check in all contours
for cnt in contours:
    idx += 1
    x, y, w, h = cv2.boundingRect(cnt)
    roi = img[y:y + h, x:x + w]
    #store my cards using certain dimensions
    if h in HANDHEIGHT and w in HANDWIDTH: 
        cv2.imwrite('hand/hand.png', roi)
        filename.append('hand/'+str(idx) + '.png')
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 2)
    #find table cards
    if (65<h<70 and 35<w<40): 
        cv2.imwrite('imgs/'+str(idx) + '.png', roi)
        filename.append('imgs/'+str(idx) + '.png')
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 2)
    #find how many players are on the table
    elif (h >80): 
        player +=1
        print('found player')
    
print('filenames are:', filename)


l= 0
bestMatch = [None]*5
card = [None]*5
y=0
x=0
h=68
w=38
i=0


# FIND CARDS ON THE TABLE
#go thru saved images
for imgs in os.listdir('imgs/'):
    imageB = cv2.imread('imgs/'+imgs)
    l=0.9
    #go thru sample data and
    for file in os.listdir('/newimgdata'):
        imageA = cv2.imread('newimgdata/'+file)
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        try:
            (score, diff) = compare_ssim(grayA, grayB, full=True)
        except:
            grayB=grayB[y:y+h,x:x+w]
            (score, diff) = compare_ssim(grayA, grayB, full=True)
        if score>l:
            l = score
            if file in bestMatch:
                continue
            bestMatch[i]=file
            card[i]=imgs
    i+=1



# FIND MY HAND
x=0
y=0
h=45
w=26

l = l2 = 0
BESTHANDMATCH = BESTHANDMATCH2 =''
for imgs in os.listdir('newimgdata'):
    x=0
    dataImg = cv2.imread('newimgdata/'+imgs)
    image = cv2.imread('hand/hand.png') 
    dataImg = dataImg[y:y+h,x:x+w] 
    x=24 
    try:
        imageA = image[y:y+h,x:x+w]
        x=69
        imageB = image[y:y+h,x:x+w]
        dataImg = cv2.cvtColor(dataImg, cv2.COLOR_BGR2GRAY)
        imageA = cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
        imageB = cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(imageA, dataImg, full=True)
        #Find highest match score
        if score>l:
            BESTHANDMATCH = imgs
            l=score
        (score2, diff) = compare_ssim(imageB, dataImg, full=True)
        if score2>l2:
            BESTHANDMATCH2 = imgs
            l2=score2
    except:
        pass


print('---------------------------')
print('Your hand is: {} {}'.format(BESTHANDMATCH,BESTHANDMATCH2))
print('---------------------------')
print('the cards on the table are: {}'.format(bestMatch))
print('---------------------------')
print('players playing: {}'.format(player))

    
cv2.waitKey()
cv2.destroyAllWindows()
