import numpy as np
np.random.seed(1337)  # for reproducibility
 
from keras.datasets import mnist
from keras.models import Model #泛型模型
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
import sys
import socket 
from PyQt5.QtWidgets import QApplication , QMainWindow
#class Ui_MainWindow(QtWidgets.QMainWindow):  #用这个替换Ui_Mind_locker_Ui.py 的 class Ui_MainWindow(object):

from Ui_Mind_Locker_Ui import *
import numpy as np
import random
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QFileInfo
#import qdarkstyle
import keras
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Dropout,Activation,regularizers
from keras.layers import Embedding, LSTM,SimpleRNN, Reshape
from keras.optimizers import RMSprop,Adam
from keras.utils import np_utils
from keras.callbacks import TensorBoard,EarlyStopping
import matplotlib.pyplot as plt

####################  全局超参数 ##############################
total_EEG_data_number=1000 #读取n条EEG数据 注意这个数字必须是time_steps 100的倍数
total_EEG_Features=18  #这是固定的。每一条EEG数据都有24个参数值。
training_times=100 #训练的次数
training_batch_size=100 #每次训练输入的EEG帧数
total_EEG_group_number_for_test=10 #每次检测所采样的EEG帧组数
server_address='127.0.0.1'
step=1 #此参数是：action（）的操作的步骤的标志。
filename=''
directly_load_filename=''
match_triger=0.9  #此参数设置了每一帧的通过测试的阀门值。
total_EEG_data_number_times=20  #此参数设置了EEG采样的倍数。比如中间值为12*50,注意，因为两个陪训的数据集都只有2千条，所以采样的EEG也最多只能2千条，所以这个数字只能设置为最大19.
directly_load_model_flag=False  #此参数是：是否直接读取预训练模型的标志。
directly_load_EEG_flag=False #此参数是：是否直接读取预录制EEG的标志
time_steps=1  #明天写注释！！！！

f = open('ljz_LSTM_2EEG.txt', 'r') #这是用于参与训练的他人EEG
All_EEG_data_lines=f.readlines()
EEG_data_B=np.zeros([total_EEG_data_number,total_EEG_Features])
#region  
for k in range(total_EEG_data_number):
    EEG_data_one_line=(All_EEG_data_lines[k].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份
    for i in range(total_EEG_Features):
        if len(EEG_data_one_line)==total_EEG_Features+1: #这个判断是为了避免有时候读取EEG时候，遇到换行符丢失的现象。
            EEG_data_B[k][i]=float(EEG_data_one_line[i])
        else:
            EEG_data_B[k][i]=EEG_data_B[k-1][i]
f.close()


print(type(EEG_data_B[0][0]))
#endregion
print('max=',EEG_data_B.max())
EEG_data_B = EEG_data_B.astype('float32') / 20      # minmax_normalized

EEG_data_B=EEG_data_B.reshape(int(total_EEG_data_number/time_steps),int(time_steps*total_EEG_Features))
# x_train=EEG_data_B[:int(total_EEG_data_number/time_steps*0.75)]
# x_test=EEG_data_B[int(total_EEG_data_number/time_steps*0.75):]
x_train=EEG_data_B
# 压缩特征维度至2维
encoding_dim = 10
 
# this is our input placeholder
input_img = Input(shape=(int(time_steps*total_EEG_Features),))
 
# 编码层
encoded = Dense(160, activation='relu',activity_regularizer=regularizers.l1(0))(input_img)  #10e-7
encoded = Dense(80, activation='relu',activity_regularizer=regularizers.l1(0))(encoded)
encoded = Dense(40, activation='relu',activity_regularizer=regularizers.l1(0))(encoded)

encoded = Dense(20, activation='relu',activity_regularizer=regularizers.l1(0))(encoded)

encoder_output = Dense(encoding_dim)(encoded)
 
# 解码层
decoded = Dense(20, activation='relu',activity_regularizer=regularizers.l1(0))(encoder_output)

decoded = Dense(40, activation='relu',activity_regularizer=regularizers.l1(0))(decoded)

decoded = Dense(80, activation='relu',activity_regularizer=regularizers.l1(0))(decoded)
decoded = Dense(160, activation='relu',activity_regularizer=regularizers.l1(0))(decoded)
decoded = Dense(int(time_steps*total_EEG_Features), activation='tanh')(decoded)
 
# 构建自编码模型
autoencoder = Model(inputs=input_img, outputs=decoded)
 
# 构建编码模型
encoder = Model(inputs=input_img, outputs=encoder_output)
 
# compile autoencoder
#adam=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
autoencoder.compile(optimizer='adam', loss='mse')
encoder.compile(optimizer='adam', loss='mse') 
# training
print(x_train.shape)
early_stopping = EarlyStopping(monitor='loss',patience=int(training_times*0.2),verbose=1,mode='auto') 

autoencoder.fit(x_train, x_train, epochs=training_times, batch_size=100, shuffle=True,callbacks=[early_stopping])
 
# plotting

encoder.save('ljz_encoder.h5')
autoencoder.save('ljz_LSTM_2_key.h5')
x = autoencoder.predict(EEG_data_B.reshape(int(total_EEG_data_number/time_steps),int(time_steps*total_EEG_Features)))

# print(x)
# print(EEG_data_B)
print('original-X=',sum(abs(EEG_data_B-x)))
print('Total original-X=',sum(abs(sum(abs((EEG_data_B-x))))))
# # add 
# model=Sequential()

# model.add(LSTM(300,activation='softsign',input_shape = (time_steps, total_EEG_Features),dropout=0.2,recurrent_dropout=0.1, stateful=False,return_sequences=False))  #stateful=True,可以使得帧组之间产生关联。 记得要在fit时候，shuffle=True。

# model.add(Dense(300,kernel_regularizer=regularizers.l2(0.02),bias_regularizer=regularizers.l2(0.02)))
# model.add(Activation('relu'))
# model.add(Dropout(0.4))

# model.add(Dense(300,kernel_regularizer=regularizers.l2(0.02),bias_regularizer=regularizers.l2(0.02)))
# model.add(Activation('relu'))
# model.add(Dropout(0.4))

# model.add(Dense(300,kernel_regularizer=regularizers.l2(0.02),bias_regularizer=regularizers.l2(0.02)))
# model.add(Activation('relu'))
# model.add(Dropout(0.4))

# model.add(Dense(1))
# model.add(Activation('sigmoid'))

# rmsprop=RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
# adam=Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['binary_accuracy'])
# #model.compile(loss='binary_crossentropy',optimizer=rmsprop,metrics=['binary_accuracy'])

# ########################神经网络搭建完毕#############################


# ################这个tb，是为了使用TensorBoard########################
# tb = TensorBoard(log_dir='./logs',  # log 目录
#     histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算. 好像只能设置为0，否则程序死机。
#     batch_size=32,     # 用多大量的数据计算直方图
#     write_graph=True,  # 是否存储网络结构图
#     write_grads=False, # 是否可视化梯度直方图
#     write_images=False,# 是否可视化参数
#     embeddings_freq=0, 
#     embeddings_layer_names=None, 
#     embeddings_metadata=None)    

# # 在命令行，先conda activate envs，然后进入本代码所在的目录，然后用 tensorboard --logdir=logs/ 来看log
# # 然后打开chrome浏览器，输入http://localhost:6006/ 来查看
# # 如果出现tensorboard错误，那么需要修改 ...\lib\site-packages\tensorboard\manager.py，其中keras环境下的这个文件，我已经修改好了。
# ########################开始训练#############################
# #training_loop_times=100  # 把进度条分为10分，所以训练也分解为 10次。
# # for i in range(training_loop_times):  #这个for，只是为了进度条的显示，所以分成 10次来训练。
# #     per_step_result=model.fit(x, y,validation_split=0.33, epochs=int(max(training_times/training_loop_times,1)), batch_size=training_batch_size,verbose = 1,shuffle=True) #这一行没有带callbacks，所以无法使用TensorBoard
# #     final_result_loss=str(per_step_result.history['loss'][0])[:5]
# #     final_result_acc=str(per_step_result.history['acc'][0])[:5]
# #     print("Training loop times:",i,"/100")
# #     ui.label.setText("开始机器学习你的脑纹。目前的损失率为"+final_result_loss+"  目前的准确率为"+final_result_acc)
# #     ui.progressBar.setProperty("value", (i+1)*100/training_loop_times)
# #     QApplication.processEvents()  #用于PyQt界面的刷新，保证流畅程度。
# early_stopping = EarlyStopping(monitor='val_loss',patience=int(training_times*0.1),verbose=1,mode='min') 

# x = x.reshape((-1, time_steps, total_EEG_Features))
# y = np.ones([int(total_EEG_data_number/time_steps),1]) 
# model.fit(x, y, validation_split=0.33,epochs=training_times, batch_size=training_batch_size,verbose = 1,shuffle=False,callbacks=[early_stopping]) #这一行带callbacks，是为了使用TensorBoard
# autoencoder.save('ljz_autoencoder.h5')
# model.save('ljz_LSTM.h5')


# # plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=1, s=3)
# # plt.colorbar()
# # plt.show()

# # print (encoded_imgs)
# #####################################################  训练完毕 #############################################
# # ruanjiyang-AEEG.txt 
# # ljz_LSTMEEG.txt

# f = open('ljz_LSTMEEG.txt', 'r') #这是用于参与训练的他人EEG
# All_EEG_data_lines=f.readlines()
# EEG_data_B=np.zeros([total_EEG_data_number,total_EEG_Features])

# for k in range(total_EEG_data_number):
#     EEG_data_one_line=(All_EEG_data_lines[k].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份
#     for i in range(total_EEG_Features):
#         if len(EEG_data_one_line)==total_EEG_Features+1: #这个判断是为了避免有时候读取EEG时候，遇到换行符丢失的现象。
#             EEG_data_B[k][i]=float(EEG_data_one_line[i])
#         else:
#             EEG_data_B[k][i]=EEG_data_B[k-1][i]
# f.close()

# x=autoencoder.predict(EEG_data_B)
# x = x.reshape((-1, time_steps, total_EEG_Features))
# result= model.predict(x,verbose = 1)
# print(result)