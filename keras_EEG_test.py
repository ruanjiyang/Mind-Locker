import numpy as np
import random
import keras
#import matplotlib.pyplot as plt
#%matplotlib inline  #将matplotlib图片直接潜入到notebook中。

from keras.datasets import mnist
from keras.models import Sequential, Model,load_model
from keras.layers.core import Dense, Dropout,Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

total_EEG_data_number=1000#读取n条EEG数据
total_EEG_Features=24  #这是固定的。每一条EEG数据都有24个参数值。
training_times=500 #训练的次数
learn_rate=0.02 #学习率

model=load_model('trained Mind Locker.h5')


f = open('who_EEG.txt', 'r')
All_EEG_data_lines=f.readlines()
EEG_data_for_test=np.zeros([total_EEG_data_number,total_EEG_Features])

for k in range(total_EEG_data_number):
    EEG_data_one_line=(All_EEG_data_lines[k].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份
    # print(EEG_data_one_line)
    for i in range(total_EEG_Features):
        EEG_data_for_test[k][i]=float(EEG_data_one_line[i])
f.close()


test=model.predict(EEG_data_for_test[:total_EEG_data_number],verbose = 1)
print(test)

match_counter=0

for k in range(total_EEG_data_number):
    if test[i]>=0.8:
        match_counter+=1
print(match_counter)
print("total match rate=", match_counter/total_EEG_data_number*100,"%")





