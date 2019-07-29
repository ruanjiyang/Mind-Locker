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
total_EEG_Features=18  #这是固定的。每一条EEG数据都有24个参数值。
training_times=500 #训练的次数
training_batch_size=200 #每次训练输入的EEG帧数
total_EEG_number_for_test=10 #每次检测所采样的EEG帧组数
server_address='127.0.0.1'
step=1 #此参数是：action（）的操作的步骤的标志。
filename=''
directly_load_filename=''
match_triger=0.9  #此参数设置了每一帧的通过测试的阀门值。
total_EEG_number_for_test_times=20  #此参数设置了EEG采样的倍数。比如中间值为12*50,注意，因为两个陪训的数据集都只有2千条，所以采样的EEG也最多只能2千条，所以这个数字只能设置为最大19.
directly_load_model_flag=False  #此参数是：是否直接读取预训练模型的标志。
directly_load_EEG_flag=False #此参数是：是否直接读取预录制EEG的标志
time_steps=1  #明天写注释！！！！


 
# # X shape (60,000 28x28), y shape (10,000, )
# (x_train, _), (x_test, y_test) = mnist.load_data()
 
# # 数据预处理
# x_train = x_train.astype('float32') / 255. - 0.5       # minmax_normalized
# x_test = x_test.astype('float32') / 255. - 0.5         # minmax_normalized
# x_train = x_train.reshape((x_train.shape[0], -1))
# x_test = x_test.reshape((x_test.shape[0], -1))
# print(x_train.shape)
# print(x_test.shape)
 

autoencoder=load_model('ljz_LSTM_2_key.h5')


# plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=1, s=3)
# plt.colorbar()
# plt.show()

# print (encoded_imgs)
#####################################################  训练完毕 #############################################
# ruanjiyang-AEEG.txt 1_EEG.txt  rjh_LSTM_EEG.txt  ruanjingheng_LSTM_TS_100EEG.txt
#ljz_LSTMEEG.txt   ljz_LSTM_2EEG.txt   ljz_LSTM_100EEG.txt

f = open('ljz_LSTM_2EEG.txt', 'r') #这是用于参与训练的他人EEG
All_EEG_data_lines=f.readlines()
EEG_data_B=np.zeros([total_EEG_number_for_test,total_EEG_Features])
final_score=0
max_score=0
for test_times in range (int(2000/total_EEG_number_for_test)):  #0,1,2,3....
    for k in range(total_EEG_number_for_test):
        EEG_data_one_line=(All_EEG_data_lines[k+test_times*total_EEG_number_for_test].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份
        for i in range(total_EEG_Features):
            if len(EEG_data_one_line)==total_EEG_Features+1: #这个判断是为了避免有时候读取EEG时候，遇到换行符丢失的现象。
                EEG_data_B[k][i]=float(EEG_data_one_line[i])
            else:
                EEG_data_B[k][i]=EEG_data_B[k-1][i]
    f.close()
    EEG_data_B = EEG_data_B.astype('float32') / 20     # minmax_normalized

    EEG_data_B=EEG_data_B.reshape(int(total_EEG_number_for_test/time_steps),int(time_steps*total_EEG_Features))
    x=autoencoder.predict(EEG_data_B)
    print('Total original-X=',sum(abs(sum(abs((EEG_data_B-x))))))
    final_score+=sum(abs(sum(abs((EEG_data_B-x)))))
    max_score=max(max_score,sum(abs(sum(abs((EEG_data_B-x))))))
# result= model.predict(x,verbose = 1)
print('Final Average Score=', (final_score-max_score)/(int(2000/total_EEG_number_for_test)-1))