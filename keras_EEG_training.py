import numpy as np
import random
import keras


from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Dropout,Activation,regularizers
from keras.layers import Embedding, LSTM,SimpleRNN,Reshape
from keras.optimizers import RMSprop
from keras.utils import np_utils

####################### 从 EEG_data_A.txt 和  EEG_data_B.txt 读取 EEG数据############################

total_EEG_data_number=1000#读取n条EEG数据
total_EEG_Features=24  #这是固定的。每一条EEG数据都有24个参数值。
training_times=10 #训练的次数
learn_rate=0.02 #学习率

f = open('ruanjiyang-A-1.txt', 'r')  #这是需要设定的EEG
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

f = open('ruanjingheng-1-2c.txt', 'r') #这是用于参与训练的他人EEG
All_EEG_data_lines=f.readlines()
EEG_data_B=np.zeros([total_EEG_data_number,total_EEG_Features])

for k in range(total_EEG_data_number):
    EEG_data_one_line=(All_EEG_data_lines[k].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份

    for i in range(total_EEG_Features):
        EEG_data_B[k][i]=float(EEG_data_one_line[i])
f.close()
print(EEG_data_B[:5,])
###############读取EEG_data_B.txt完毕######################


f = open('ruanminli-2c.txt', 'r') #这也是用于参与训练的他人EEG
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

y0 = np.ones([int(total_EEG_data_number/5),1])               # class0 y data (tensor), shape=(100, 1)

y1 = np.zeros([int(total_EEG_data_number/5),1])                # class1 y data (tensor), shape=(100, 1)
y2 = np.zeros([int(total_EEG_data_number/5),1])                # class1 y data (tensor), shape=(100, 1)
x=np.vstack((EEG_data_A,EEG_data_B,EEG_data_C))
print(x.shape)
y = np.vstack((y0,y1,y2))   # shape (200,) LongTensor = 64-bit integer
print(y.shape)

model=Sequential()

# model.add(Dense(100, input_shape=(24,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
x = x.reshape((int(x.shape[0]/5), 5, x.shape[1]))
print("x shape=",x.shape)
#model.add(Reshape((1,24)))
model.add(LSTM(100,activation='relu',input_shape = (5, 24),dropout=0.2,recurrent_dropout=0.2, return_sequences=False))
# model.add(Reshape((-1,24)))
# model.add(SimpleRNN(100,return_sequences=False))


# model.add(Dense(100))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))


model.add(Dense(1))
model.add(Activation('sigmoid'))

#print(model.summary())

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

history = model.fit(x, y, epochs=200, batch_size=200, verbose = 1,shuffle=False)

model.save('trained Mind Locker-ruanjiyang.h5')





