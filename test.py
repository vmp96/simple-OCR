#extracts features from "test.bmp", normalizes based on mean and standard deviation stored in nbd.dat
#creates distance matrix between test features and training data features
#called after recognize.py

import numpy as np
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import math
import os

def extract(img_name, plot):
    img = io.imread(img_name)
    if plot:
        print img.shape
        io.imshow(img)
        plt.title('Original Image')
        io.show()
        

    if plot:
        hist=exposure.histogram(img)
        plt.bar(hist[1],hist[0])
        plt.title('Histogram')
        plt.show()

    th=211
    img_binary= (img<th).astype(np.double)

    #"""
    if plot:
        io.imshow(img_binary)
        plt.title('Binary Image')
        io.show()
    #"""

    img_label = label(img_binary,background=0)
    
    if plot:
        io.imshow(img_label)
        plt.title('Labeled Image')
        io.show()
        print np.amax(img_label)
        
    regions = regionprops(img_label)
    if plot:
        io.imshow(img_binary)
        ax=plt.gca()
        plt.title('Bounding Boxes')
    
    
    Features = []
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        area=(maxr-minr)*(maxc-minc)
        if area<150 or maxr-minr<=9 or maxc-minc<=9 or area>10000:
            continue
        roi=img_binary[minr:maxr, minc:maxc]
        m=moments(roi)
        cr = m[0,1]/m[0,0]
        cc = m[1,0]/m[0,0]
        mu=moments_central(roi,cr,cc)
        nu=moments_normalized(mu)
        hu=moments_hu(nu)
        """
        white=0.0
        for i in range(maxr-minr):
            for j in range(maxc-minc):
                if roi[i][j]==1:
                    white+=1
        white=white/area
        hu=np.append(hu,[white])
        """
        Features.append(hu)
        if plot:
            ax.add_patch(Rectangle((minc,minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1 ))
    if plot:
        io.show()
   
    return Features


def char_value(char_list, value):
    for i in range(len(char_list)):
        if value<=char[i][1]:
            return char[i][0]
    return 0

def normalize(Features, mean, std):
    std=std*1.0
    
    for i in range(len(Features)): #characters
        for j in range (7): #features
            Features[i][j]=(Features[i][j]-mean)/std
        #ndb[i][1][j][7]=ndb[i][1][j][7]/100.0
                 
Test_Features=extract('test1.bmp',True)

with open('ndb.dat', 'rb') as rfp:
    ndb=pickle.load(rfp)
    
normalize(Test_Features,ndb[len(ndb)-1][0],ndb[len(ndb)-1][0])

#create 2D feature matrix
Features=[]
for i in range(len(ndb)-1): #characters
        for j in range(len(ndb[i][1])): #regions
            Features.append(ndb[i][1][j])

char=[]
total=0
for i in range(len(ndb)-1):
    total+=len(ndb[i][1])
    if i==0:
        total=total-1
    char_index=[ndb[i][0],total]
    char.append(char_index)


D=cdist(Test_Features,Features)

io.imshow(D)
plt.title('Distance Matrix')
io.show()

#find smallest index in each row of D without sort
D_index=[]
for i in range(len(D)):
    min=1000
    index=-1
    for j in range(len(D[i])):
        if D[i][j]<min:
            min=D[i][j]
            index=j
    D_index.append(index)


Ytrue = []
Ypred = []
for i in range(len(D_index)):
   # Ytrue.append(char_value(char,i))
    Ypred.append(char_value(char,D_index[i]))
    #print(D_index[i]),
    #print(Ypred[i]),
   
