#Normalizes features in "db.dat" and creates distance and confusion matrices
#called after train.py 

import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import math
import os

def char_value(char_list, value):
    for i in range(len(char_list)):
        if value<=char[i][1]:
            return char[i][0]
    return 0

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
            #ndb[i][1][j][7]=ndb[i][1][j][7]/100.0
    
    ndb.append([mean,std])
    
    with open(fname, 'wb') as wfp:
        pickle.dump(ndb,wfp)
    
#"""
with open('db.dat', 'rb') as rfp:
    db=pickle.load(rfp)
normalize('ndb.dat',db)
#"""

with open('ndb.dat', 'rb') as rfp:
    ndb=pickle.load(rfp)


#create 2D feature matrix
Features=[]
for i in range(len(ndb)-1): #characters
        for j in range(len(ndb[i][1])): #regions
            Features.append(ndb[i][1][j])


#get absolute character range index for Features
char=[]
total=0
for i in range(len(ndb)-1):
    total+=len(ndb[i][1])
    if i==0:
        total=total-1
    char_index=[ndb[i][0],total]
    char.append(char_index)
   
D=cdist(Features,Features)

io.imshow(D)
plt.title('Distance Matrix')
plt.savefig('Distance_matrix.png', bbox_inches='tight')
io.show()


#find second smallest index in each row of D without sort
D_index=[]
for i in range(len(D)):
    min=1000
    index=-1
    for j in range(len(D[i])):
        if i!=j and D[i][j]<min:
            min=D[i][j]
            index=j
    D_index.append(index)

Ytrue = []
Ypred = []
count=0.0
for i in range(len(D_index)):
    Ytrue.append(char_value(char,i))
    Ypred.append(char_value(char,D_index[i]))
   
 
confM=confusion_matrix(Ytrue, Ypred)


io.imshow(confM)
plt.title('Confusion Matrix')
plt.savefig('Confusion_matrix.png', bbox_inches='tight')
io.show()


