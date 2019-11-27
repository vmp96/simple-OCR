#Takes console input and adds given label and file to features database
#database stored in file "db.dat"

import numpy as np
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import os

def train(symbol, img_name, plot):
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
    #io.imsave(symbol+"_binary.jpg", img_binary, None, quality=100)

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
        io.imshow(img_label)
        ax=plt.gca()
        plt.title('Bounding Boxes')
    
    db=[]
    if not os.path.exists('db.dat'):
        print('Creating database')
    else:
        with open('db.dat', 'rb') as rfp:
            db=pickle.load(rfp)
    
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
        #find percentage of bb taken up by character
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
    data=[symbol, Features]
    db.append(data)
    with open('db.dat', 'wb') as wfp:
        pickle.dump(db,wfp)
    
        
char=raw_input('Input character:')
while len(char)!=1 or not char.isalpha():
    char=raw_input('Input must be single alphabetic character:')
fname=raw_input('Input training file:')
plot=raw_input('Display plots? (y/n):')
while plot!='y' and plot!='n':
    plot=raw_input('Display plots? Please enter y/n:')

if plot=='y':
    plot=True
else:
    plot=False

train(char,fname,plot)
print("Training data saved to db.dat")
