import numpy as np
import glob
import pylab as plt
import csv
import librosa as lib
import pandas as pd
number=[]
f_file=open('testData.csv','w')
for f in glob.glob('E:/Work/B.E Project/TensorFlow Speech Recognition Challenge/test/audio/*.wav'):
    y,sr=lib.load(f)
    features=lib.feature.mfcc(y,sr,n_mfcc=20).T
    mfcc=list(np.mean(features, axis=0))
    std_col= list(np.std(features, axis=0)) 
    mfcc.extend(std_col)
    f= f.strip('.wav')
    files= f.split('-')   
    f,lab=f.split('\\')
    print(lab)
    
    for i in files:
        f_file.write('{i},'.format(i=lab))
    for mf in mfcc:
        f_file.write('{mf},'.format(mf=mf))
        
    f_file.write('\n')
        