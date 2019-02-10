import numpy as np
import glob
import pylab as plt
import csv
import librosa as lib
import pandas as pd
number=[]
folders = glob.glob('E:/Work/B.E Project/TensorFlow Speech Recognition Challenge/test/audio/*')
f_file=open('testData.csv','w')
for folder in folders:
    print(folder)
    for f in glob.glob(folder+'/*.wav'):
        y,sr=lib.load(f)
        features=lib.feature.mfcc(y,sr,n_mfcc=20).T
        mfcc=list(np.mean(features, axis=0))
        std_col= list(np.std(features, axis=0)) 
        mfcc.extend(std_col)
        f= f.strip('.wav')
        files= f.split('-')   
        f,lab=folder.split('\\')
        print(lab)
#pd.df.insert(0,'labels',lab)
        for i in files:
            f_file.write('{i},'.format(i=lab))
        for mf in mfcc:
            f_file.write('{mf},'.format(mf=mf))
        #f_file.write('{feat}\t{label}'.format(feat=mfcc,label=lab))
            #write('{feat}\t{label}'.format(feat = mfccs, label=label))
        f_file.write('\n')
          
        ##try:
          #  X, sample_rate = librosa.load(f,res_type='kaiser_fast') 
          #  mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
        #except Exception as e:
         #   print("Error encountered while parsing file: ",f)
  
