#Does everything
#takes test file via console input
#Overwrites any current db.dat or ndb.dat
#Creates these files similar to the methods used in train and recognize with the training files a.bmp, d.bmp...
#Attempts ocr with training data on the test file
#Prints all plots including the final bounding box image with accompanying ocr results

import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import math

#sort that puts in order from top to bottom, left to right
def sort(Features):
    sortList = []
    unsort=Features[:]
    for i in range(len(unsort)):
        minc=100000
        minr=100000
        index=-1
        for j in range (len(unsort)):
            if unsort[j][8]<minr and abs(unsort[j][8]-minr)>30:
                minr=unsort[j][8]
                minc=unsort[j][7]
                index=j
            else: 
                if abs(unsort[j][8]-minr)<10 and unsort[j][7]<minc:
                    minr=unsort[j][8]
                    minc=unsort[j][7]
                    index=j
        sortList.append(unsort.pop(index))
    return sortList
            
            
def train(symbol, img_name, plot,db=None):
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
    #img_binary=io.imread('test a binary fix1.bmp')

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

    if not db:
        db = []
    
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
        plt.savefig(symbol+'_component_bb.png', bbox_inches='tight', trasparent=False)
        io.show()
        #io.imsave(symbol+"_component_bb.jpg", img_label, None, quality=100)
    data=[symbol, Features]
    db.append(data)
    return db

def autotrain():
    chars = ['d','m','n','o','p','q','r','u','w']  
    db=train('a','a.bmp',True)     
    for i in range(len(chars)):
        db=train(chars[i],chars[i]+".bmp",True,db)
    with open('db.dat', 'wb') as wfp:
        pickle.dump(db,wfp)
    

def normalize(fname, db):
    count=0
    total=0
    for i in range(len(db)): #characters
        for j in range(len(db[i][1])): #regions
            for k in range (7): #features
                total+=db[i][1][j][k]
                count+=1
                
    mean=total/count
    
    ndb=db[:]
    total=0
    for i in range(len(db)): #characters
        for j in range(len(db[i][1])): #regions
            for k in range (7): #features
                ndb[i][1][j][k]= ndb[i][1][j][k]-mean
                total+=math.pow(ndb[i][1][j][k],2)
    
    
    std=math.sqrt(total/count)
    
    for i in range(len(db)): #characters
        for j in range(len(db[i][1])): #regions
            for k in range (7): #features
                ndb[i][1][j][k]= ndb[i][1][j][k]/std
            #ndb[i][1][j][7]=ndb[i][1][j][7]/1.0
    
    ndb.append([mean,std])
   
    with open(fname, 'wb') as wfp:
        pickle.dump(ndb,wfp)

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
    #io.imsave(symbol+"_binary.jpg", img_binary, None, quality=100)

    #"""
    if plot:
        io.imshow(img_binary)
        plt.title('Binary Image')
        io.show()
    #"""

    img_label = label(img_binary,background=0)
    
    if plot:
        io.imshow(img_binary)
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
        hu=np.append(hu,[cc+minc,cr+minr])
        Features.append(hu)
       
        if plot:
            ax.add_patch(Rectangle((minc,minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1 ))
    if plot:
        #io.imsave("test_binary.jpg", img_label, None, quality=100)
        #plt.savefig('test_component_bb.png', bbox_inches='tight', trasparent=False)
        io.show()
   
    return Features


def char_value(char_list, value):
    for i in range(len(char_list)):
        if value<=char_list[i][1]:
            return char_list[i][0]
    return 0

def normalizeGiven(Features, mean, std):
    std=std*1.0
    
    for i in range(len(Features)): #characters
        for j in range (7): #features
            Features[i][j]=(Features[i][j]-mean)/std
        #Features[i][7]=Features[i][7]/1.0

def ocr(fname):
    #train and normalize with training files a.bmp ...
    autotrain()
    with open('db.dat', 'rb') as rfp:
        db=pickle.load(rfp)
    normalize('ndb.dat',db)
    #get features from given file
    CTest_Features=extract(fname,True)
    CTest_Features=sort(CTest_Features)
    
    with open('ndb.dat', 'rb') as rfp:
        ndb=pickle.load(rfp)
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

    Test_Features=[]
    for i in range(len(CTest_Features)):
        Test_Features.append(CTest_Features[i][0:7])
    
       
    
    normalizeGiven(Test_Features,ndb[len(ndb)-1][0],ndb[len(ndb)-1][0])
    D=cdist(Test_Features,Features)

    io.imshow(D)
    plt.title('Distance Matrix')
    #plt.savefig('Distance_Matrix.png', bbox_inches='tight', trasparent=False)
    io.show()

    #find 5 smallest indeces in each row of D without sort and store most common
    D_index=[]
    
    
    for i in range(len(D)):
        min=1000
        index=-1
        for j in range(len(D[i])):
            if D[i][j]<min:
                min=D[i][j]
                index=j
        D_index.append(index)
    
    """
    for i in range(len(D)):
        min=[1000,1000,1000,1000,1000]
        index=[-1,-1,-1,-1,-1]
        for j in range(len(D[i])):
            if D[i][j]<min[0]:
                min[4]=min[3]
                min[3]=min[2]
                min[2]=min[1]
                min[1]=min[0]
                min[0]=D[i][j]
                index[4]=index[3]
                index[3]=index[2]
                index[2]=index[1]
                index[1]=index[0]
                index[0]=j
            else: 
                if D[i][j]<min[1]:
                    min[4]=min[3]
                    min[3]=min[2]
                    min[2]=min[1]
                    min[1]=D[i][j]
                    index[4]=index[3]
                    index[3]=index[2]
                    index[2]=index[1]
                    index[1]=j
                else: 
                    if D[i][j]<min[2]:
                        min[4]=min[3]
                        min[3]=min[2]
                        min[2]=D[i][j]
                        index[4]=index[3]
                        index[3]=index[2]
                        index[2]=j
                    else: 
                        if D[i][j]<min[3]:
                            min[4]=min[3]
                            min[3]=D[i][j]
                            index[4]=index[3]
                            index[3]=j
                        else: 
                            if D[i][j]<min[4]:
                                min[4]=D[i][j]
                                index[4]=j
        #end inner loop
        for k in range(5):
            index[k]=char_value(char,index[k])
        #find mode (weight on distance, no sort)
        freq = []
        for k in range(5):
            if index[k] not in freq:
                count = 0
                for j in range(5):
                    if index[j]==index[k]:
                        count+=1
                freq.append([index[k],count])
        #find max frequency 
        max=0
        for k in range(len(freq)):
            if freq[k][1]>max:
                max=freq[k][1]
                label=freq[k][0]
        
        D_index.append(label)
                #print(D_index[i]),
       
    #"""
    pkl_file=open('test1_gt.pkl.txt', 'rb')
    mydict=pickle.load(pkl_file)
    pkl_file.close()
    classes = mydict['classes']
    locations = mydict['locations']
    
    
    
    #char_value(D_index[i]) returns detected character such that i is the training label Features[i]
    #CTest_Features[i][7] contains 2*1 list with [meanc, meanr] representing center of the same label Features[i]
    total=0
    Feature_Center = []
    for i in range(len(CTest_Features)):
        Feature_Center.append(CTest_Features[i][7:9])
    D=cdist(locations,Feature_Center)
    Results=[]
    for i in range (len(locations)):
        for c in range (len(locations[i])):
            min=10000
            index=-1
            for j in range(len(D_index)):
                if D[i][j]<min:
                    min=D[i][j]
                    index=j
        Results.append([classes[i],char_value(char,D_index[index])])
        """
        print("Expected "+classes[i]),
        print("Detected"),
        print(char_value(char,D_index[index])),
        print(D_index[index])
        """
    return Results

fname=raw_input('Input testing file:')
Results=ocr(fname)
img = io.imread(fname)
th=211
img_binary= (img<th).astype(np.double)
img_label = label(img_binary,background=0)
io.imshow(img_binary)
ax=plt.gca()
plt.title('Bounding Boxes')
regions = regionprops(img_label)
for props in regions:
    minr, minc, maxr, maxc = props.bbox
    ax.add_patch(Rectangle((minc,minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1 ))
io.show()
print("From top left to bottom right")
for i in range(len(Results)):
    print("Expected "),
    print(Results[i][0]),
    print("Detected"),
    print(Results[i][1])
