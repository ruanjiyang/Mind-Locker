import numpy as np
import random
import keras
#import matplotlib.pyplot as plt
#%matplotlib inline  #将matplotlib图片直接潜入到notebook中。

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout,Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

####################### 从 EEG_data_A.txt 和  EEG_data_B.txt 读取 EEG数据############################

total_EEG_data_number=1000#读取n条EEG数据
total_EEG_Features=24  #这是固定的。每一条EEG数据都有24个参数值。
training_times=500 #训练的次数
learn_rate=0.02 #学习率

f = open('EEG_data_Ruanjiyang.txt', 'r')
All_EEG_data_lines=f.readlines()
EEG_data_A=np.zeros([total_EEG_data_number,total_EEG_Features])

for k in range(total_EEG_data_number):
    EEG_data_one_line=(All_EEG_data_lines[k].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份
    # print(EEG_data_one_line)
    for i in range(total_EEG_Features):
        EEG_data_A[k][i]=float(EEG_data_one_line[i])
f.close()
print(EEG_data_A[:5,])
###############读取EEG_data_A.txt完毕######################

f = open('EEG_data_Ruanminli.txt', 'r')
All_EEG_data_lines=f.readlines()
EEG_data_B=np.zeros([total_EEG_data_number,total_EEG_Features])

for k in range(total_EEG_data_number):
    EEG_data_one_line=(All_EEG_data_lines[k].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份

    for i in range(total_EEG_Features):
        EEG_data_B[k][i]=float(EEG_data_one_line[i])
f.close()
print(EEG_data_B[:5,])
###############读取EEG_data_B.txt完毕######################


f = open('EEG_data_Ruanjingheng.txt', 'r')
All_EEG_data_lines=f.readlines()
EEG_data_C=np.zeros([total_EEG_data_number,total_EEG_Features])

for k in range(total_EEG_data_number):
    EEG_data_one_line=(All_EEG_data_lines[k].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份

    for i in range(total_EEG_Features):
        EEG_data_C[k][i]=float(EEG_data_one_line[i])
f.close()

###############读取EEG_data_B.txt完毕######################



###########################从 OPENBCI读取 EEG数据结束##########################

########################开始生成 数据样本
#n_data = np.ones([total_EEG_data_number, total_EEG_Features])   
# x0 = torch.normal(EEG_data_A, 20)      # class0 x data (tensor), shape=(100, 2)
y0 = np.ones([total_EEG_data_number,1])               # class0 y data (tensor), shape=(100, 1)
# x1 = torch.normal(EEG_data_B, 20)     # class1 x data (tensor), shape=(100, 2)
y1 = np.zeros([total_EEG_data_number,1])                # class1 y data (tensor), shape=(100, 1)
y2 = np.zeros([total_EEG_data_number,1])                # class1 y data (tensor), shape=(100, 1)
x=np.vstack((EEG_data_A,EEG_data_B,EEG_data_C))
print(x.shape)
y = np.vstack((y0,y1,y2))   # shape (200,) LongTensor = 64-bit integer
print(y.shape)

model=Sequential()
model.add(Dense(100, input_shape=(24,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))


model.add(Dense(1))
model.add(Activation('sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
# model.fit(x_train,y_train,epochs=10,batch_size=128,verbose=1,validation_data=[x_test,y_test])
# history = model.fit(x[:700,:], y[:700,:], epochs=100, batch_size=200,
#                    verbose = 1, validation_data=[x[700:,:], y[700:,:]])
history = model.fit(x, y, epochs=100, batch_size=200,
                   verbose = 1,shuffle=True)

model.save('trained Mind Locker.h5')


f = open('EEG_data_Ruanjingheng.txt', 'r')
All_EEG_data_lines=f.readlines()
EEG_data_for_test=np.zeros([total_EEG_data_number,total_EEG_Features])

for k in range(total_EEG_data_number):
    EEG_data_one_line=(All_EEG_data_lines[k].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份
    # print(EEG_data_one_line)
    for i in range(total_EEG_Features):
        EEG_data_for_test[k][i]=float(EEG_data_one_line[i])
f.close()


test=model.predict(EEG_data_for_test[:100])
print(test)



