
# 手动给 x,y做shuffle。
# 添加验证级

################## 项目名称： “Mind Locker：一种基于神经网络与深度学习的脑纹锁系统” ####################

################## 拟解决的关键问题 #####################
# '''
# 目前所采用的主流生物特征加密与识别，如指纹/瞳孔/人脸识别，因其易于伪造，特征值数量有限以及重复率高等因素，几乎一一被攻克，安全性受到极大挑战。比如，目前英伟达GAN人工智能的虚拟人脸生成技术，几乎攻陷了目前所有人脸识别系统。所以，在一些对安全性要求极高的应用场合（如巨额银行转账，进出军事领域等），传统生物特征加密方法的安全性，已经受到了严重的挑战。除人脸，瞳孔，指纹识别外，我们还有其他更为先进与安全的生物特征加密手段么？ 
# 答案是：大脑。每个人的大脑，其生物特征都是真正意义上的独一无二。利用脑电波的生物特征作为一种加密手段，这就是我现在所设计的一种基于深度学习/神经网络的脑纹加密与识别技术-“Mind Locker"，它具备如下特征：

# 1.采用全连接深度学习神经网络，架构简洁，训练时长短，从脑电波EEG采样到训练并设置完毕仅需时2~3分钟。每次识别验证则仅需时8~10秒。
# 2.采样过程中，可通过想象某一画面/场景/事物等，以增加其加密复杂度。
# 3.识别正确率极高，几乎接近100%。
# 4.安全性极高，因为脑电波是持续动态变化的连续帧数据流，而非指纹/人脸等静态特征值，所以伪匹配的可能性几乎为零。


# '''
################ 主要创意点 革新点 与建议 ################
'''
脑电波EEG生物特征复杂，每一帧EEG均包含24条生物特征值（5频段的电平峰值/均值/实时值，8通道的EMG值，Focus值），且所有特征值均处于动态变化中，它们相互之间的非线性数学关系一起构成了这颗大脑的完整生物特征网。系统设计的核心在于学习与提取出这24条特征值相互间的全连接动态数学关系。如需破解，则必须完全再现此动态数学关系，且能在伪数据流的所有帧中维持此关系，所以几乎不存在破解的可能性。

借助目前神经网络的深度学习技术，我实现了让计算机去学习与提取上述数学关系，达到了99.9%以上的识别率。训练样本为实时采集的2k帧EEG数据，陪训样本为4k帧随机抽取的EEG样本库（样本库采集自学校500名同学）。神经网络结构为[input:24*240*240*240*1:output]， 训练规模为200帧*200次。隐藏层采用Relu激活函数。输出层采用Sigmoid激活函数形成逻辑回归判断。

服务端数据采集系统为OpenBCI(语言：Processing)，客户端AI框架为 Keras+Tensorflow（语言：Python）。

硬件基于OpenBCI传感器，核心为 美国TI德州仪器ADS1299采样芯片（8通道，16khz，24bits) 


'''
########################### 目前的最佳网络策略是######################
#time_steps=100
#x, y不作shuffle.
#不设置验证集。
# 以上设定，可以使得ruanjiyang无法匹配 ruanjingheng

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
training_times=500 #训练的次数
training_batch_size=200 #每次训练输入的EEG帧数
total_EEG_group_number_for_test=10 #每次检测所采样的EEG帧组数
server_address='127.0.0.1'
step=1 #此参数是：action（）的操作的步骤的标志。
filename=''
directly_load_filename=''
match_triger=0.9  #此参数设置了每一帧的通过测试的阀门值。
total_EEG_data_number_times=20  #此参数设置了EEG采样的倍数。比如中间值为12*50,注意，因为两个陪训的数据集都只有2千条，所以采样的EEG也最多只能2千条，所以这个数字只能设置为最大19.
directly_load_model_flag=False  #此参数是：是否直接读取预训练模型的标志。
directly_load_EEG_flag=False #此参数是：是否直接读取预录制EEG的标志
time_steps=10  #明天写注释！！！！

#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        print("+")
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

class Training_process_bar(keras.callbacks.Callback):
    def __init__(self):
        print('开始调用callback')
    def on_epoch_end(self, batch, logs={}):
        ui.label.setText("开始机器学习你的脑纹。目前的损失率为"+str(logs.get('loss'))+"  目前的准确率为"+str(logs.get('acc')))
        ui.progressBar.setProperty("value", (batch+1)/training_times*100)
        QApplication.processEvents()  #用于PyQt界面的刷新，保证流畅程度。

def maxminnormal(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def action(): #此函数是控制整个采样，训练， 匹配测试的过程。
    global filename,total_EEG_group_number_for_test
    global step,directly_load_model_flag,openfile_name,directly_load_filename,time_steps,directly_load_EEG_flag
    filename=ui.lineEdit.text() #写入文件名

    if step==3:   # 匹配测试
        disable_Gui()
        ui.label.setText("匹配测试中...")
        # ui.label.repaint()
        ui.pushButton.setText("匹配测试中...")   
        # ui.pushButton.repaint()

        if directly_load_model_flag==False:
            model=load_model(filename+'_key.h5')
            print("使用的是刚才训练好的模型:",filename)
        if directly_load_model_flag==True:
            model=load_model(directly_load_filename)
            print("使用的是直接读取的已经训练好的模型:", directly_load_filename)
        s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.connect((server_address,5204)) #5204 为 OPENBCI GUI的缺省服务器段的发送端口
        
        temp_total_EEG_group_number_for_test=total_EEG_group_number_for_test
        total_EEG_group_number_for_test=1 #把total_EEG_group_number_for_test赋值给temp_total_EEG_group_number_for_test后，total_EEG_group_number_for_test变为1，只是为了GUI的进度条显示测试的进度%。

        EEG_data_for_test=np.zeros([time_steps,total_EEG_Features]) # EEG_data_for_test为采样的测试EEG帧。
        match_counter=0 # 用于计数EEG通过测试的个数。
        for times in range(int(temp_total_EEG_group_number_for_test)): #这个大循环，内嵌一个1的小循环。     
            for k in range(time_steps):  #明天写注释！！！！
                EEG_data_one_line=(s.recv(1024).decode('utf-8')).split('A') ####按照字符"A"来截断每一条EEG数据，分割成24小份
                for i in range(total_EEG_Features):
                    if len(EEG_data_one_line)==total_EEG_Features+1:  #这个判断是为了避免有时候读取EEG时候，遇到换行符丢失的现象。
                        EEG_data_for_test[k][i]=float(EEG_data_one_line[i])
                    else:
                        EEG_data_for_test[k][i]=EEG_data_for_test[k-1][i]
            print("original EEG_data_for_test shape (before reshape)==",EEG_data_for_test.shape)
            #EEG_data_for_test=maxminnormal(EEG_data_for_test)  # 归一化x数据
            EEG_data_for_test=EEG_data_for_test.reshape((1, time_steps, total_EEG_Features)) #明天写注释！！！！ 
            print("EEG_data_for_test shape==",EEG_data_for_test.shape)
            test=model.predict(EEG_data_for_test,verbose = 1)
            #print("test shape ==",test.shape)
            print("test value ==",test[0][0] )
            EEG_data_for_test=EEG_data_for_test.reshape((time_steps,total_EEG_Features))#明天写注释！！！！！！！！！
            if test[0][0]>=match_triger:
                match_counter=match_counter+1
            ui.progressBar.setProperty("value", (times+1)/temp_total_EEG_group_number_for_test*100)        

            result_text="当前帧组的匹配率为："+str(test[0][0]*100)[:6]+"%"+"  整体匹配率已经达到"+str(match_counter/temp_total_EEG_group_number_for_test*100)+"%"
            ui.label.setText(result_text)
            ui.label.repaint()
            ui.lcdNumber.display(match_counter/temp_total_EEG_group_number_for_test*100)
            QApplication.processEvents()  #用于PyQt界面的刷新，保证流畅程度。
        result_text="测试结束，最终匹配率为"+str(match_counter/temp_total_EEG_group_number_for_test*100)+"%"
        ui.label.setText(result_text)
        ui.label.repaint()

        total_EEG_group_number_for_test=temp_total_EEG_group_number_for_test #重新恢复全局变量total_EEG_group_number_for_test的值。
        ui.pushButton.setText("开始匹配测试")   
        ui.pushButton.repaint()
        enable_Gui()

    if step==2:  #机器学习
        disable_Gui()
        ui.label.setText("开始机器学习你的脑纹。")
        ui.label.repaint()
        ui.pushButton.setText("2-机器学习中...")   
        ui.pushButton.repaint()


        ######################################### 开始训练 ######################################
     
        ####################### 读取代训练的EEG数据############################
        if directly_load_EEG_flag== True:
            f = open(directly_load_filename, 'r')  #读取代训练的EEG数据
            print("使用的是预先录取好的EEG文件:",directly_load_filename)
        if directly_load_EEG_flag== False:
            f = open(filename+'EEG.txt', 'r')  #读取代训练的EEG数据
            print("使用的是刚才录取的EEG文件:",filename)
        
        

        All_EEG_data_lines=f.readlines()
        EEG_data_A=np.zeros([total_EEG_data_number,total_EEG_Features])

        for k in range(total_EEG_data_number):
            EEG_data_one_line=(All_EEG_data_lines[k].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份
            for i in range(total_EEG_Features):
                if len(EEG_data_one_line)==total_EEG_Features+1:   #这个判断是为了避免有时候读取EEG时候，遇到换行符丢失的现象。
                    EEG_data_A[k][i]=float(EEG_data_one_line[i])
                else:
                    EEG_data_A[k][i]=EEG_data_A[k-1][i]
        f.close()

        ###############读取代训练的EEG数据完毕######################


        ###############开始读取代陪训的Random 1/2 EEG数据######################
        f = open('1_EEG.txt', 'r') #这是用于参与训练的他人EEG
        All_EEG_data_lines=f.readlines()
        EEG_data_B=np.zeros([total_EEG_data_number,total_EEG_Features])

        for k in range(total_EEG_data_number):
            EEG_data_one_line=(All_EEG_data_lines[k].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份
            for i in range(total_EEG_Features):
                if len(EEG_data_one_line)==total_EEG_Features+1: #这个判断是为了避免有时候读取EEG时候，遇到换行符丢失的现象。
                    EEG_data_B[k][i]=float(EEG_data_one_line[i])
                else:
                    EEG_data_B[k][i]=EEG_data_B[k-1][i]
        f.close()

        ###############读取random_EEG_1完毕######################

        f = open('2_EEG.txt', 'r') #这也是用于参与训练的他人EEG
        All_EEG_data_lines=f.readlines()
        EEG_data_C=np.zeros([total_EEG_data_number,total_EEG_Features])

        for k in range(total_EEG_data_number):
            EEG_data_one_line=(All_EEG_data_lines[k].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份
            for i in range(total_EEG_Features):
                if len(EEG_data_one_line)==total_EEG_Features+1: #这个判断是为了避免有时候读取EEG时候，遇到换行符丢失的现象。
                    EEG_data_C[k][i]=float(EEG_data_one_line[i])
                else:
                    EEG_data_C[k][i]=EEG_data_C[k-1][i]
        f.close()

        ###############读取random_EEG_2完毕######################
        f = open('3_EEG.txt', 'r') #这也是用于参与训练的他人EEG
        All_EEG_data_lines=f.readlines()
        EEG_data_D=np.zeros([total_EEG_data_number,total_EEG_Features])

        for k in range(total_EEG_data_number):
            EEG_data_one_line=(All_EEG_data_lines[k].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份
            for i in range(total_EEG_Features):
                if len(EEG_data_one_line)==total_EEG_Features+1: #这个判断是为了避免有时候读取EEG时候，遇到换行符丢失的现象。
                    EEG_data_D[k][i]=float(EEG_data_one_line[i])
                else:
                    EEG_data_D[k][i]=EEG_data_D[k-1][i]
        f.close()

        ###############读取random_EEG_3完毕######################
        ##################读取代陪训的Random 1/2 EEG数据######################

        ########################开始合成总的数据样本（包括待训练数据，以及两个陪训数据#############################
        y0 = np.ones([int(total_EEG_data_number/time_steps),1])               # class0 y data (tensor), shape=(100, 1)
        y1 = np.zeros([int(total_EEG_data_number/time_steps),1])                # class1 y data (tensor), shape=(100, 1)
        y2 = np.zeros([int(total_EEG_data_number/time_steps),1]) 
        y3 = np.zeros([int(total_EEG_data_number/time_steps),1])               # class1 y data (tensor), shape=(100, 1)

        # for i in range(int(total_EEG_data_number/time_steps)):  #把标签y标记为 -1，这是为了配合最终层的tanh输出用的。 如果最终层输出用sigmoid，那么保持y的标记为0.
        #     y1[i][0]=2
        #     y2[i][0]=-1
        #     y3[i][0]=-1

      
        x=np.vstack((EEG_data_A,EEG_data_B,EEG_data_C,EEG_data_D))
        print(x.shape)
        #y = np.vstack((y0,y1,y2,y3))   # shape (200,) LongTensor = 64-bit integer
        y=np.vstack((y0,y1,y2,y3))
        print(y.shape)

        x = x.reshape((-1, time_steps, x.shape[1]))
        #手动 打乱 x,y 次序。
        # p = np.random.permutation(range(len(x)))
        # x,y = x[p],y[p]  
        ########################开始搭建神经网络#############################
        model=Sequential()

        model.add(LSTM(100,activation='softsign',input_shape = (time_steps, total_EEG_Features),dropout=0.2,recurrent_dropout=0.1, stateful=False,return_sequences=False))  #stateful=True,可以使得帧组之间产生关联。 记得要在fit时候，shuffle=True。
        # model.add(LSTM(200,activation='tanh',input_shape = (time_steps, total_EEG_Features),dropout=0.2,recurrent_dropout=0.1, stateful=False,return_sequences=False))  #stateful=True,可以使得帧组之间产生关联。 记得要在fit时候，shuffle=True。
        
        # model.add(Dense(100,kernel_regularizer=regularizers.l2(0.002),bias_regularizer=regularizers.l2(0.002)))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.2))

        # model.add(Dense(100,kernel_regularizer=regularizers.l2(0.002),bias_regularizer=regularizers.l2(0.002)))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.2))

        # model.add(Dense(100,kernel_regularizer=regularizers.l2(0.002),bias_regularizer=regularizers.l2(0.002)))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.2))

        model.add(Dense(100,kernel_regularizer=regularizers.l2(0.002),bias_regularizer=regularizers.l2(0.002)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        rmsprop=RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
        adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])

        ########################神经网络搭建完毕#############################
        

        ################这个tb，是为了使用TensorBoard########################
        tb = TensorBoard(log_dir='./logs',  # log 目录
            histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算. 好像只能设置为0，否则程序死机。
            batch_size=32,     # 用多大量的数据计算直方图
            write_graph=True,  # 是否存储网络结构图
            write_grads=False, # 是否可视化梯度直方图
            write_images=False,# 是否可视化参数
            embeddings_freq=0, 
            embeddings_layer_names=None, 
            embeddings_metadata=None)    

        # 在命令行，先conda activate envs，然后进入本代码所在的目录，然后用 tensorboard --logdir=logs/ 来看log
        # 然后打开chrome浏览器，输入http://localhost:6006/ 来查看
        # 如果出现tensorboard错误，那么需要修改 ...\lib\site-packages\tensorboard\manager.py，其中keras环境下的这个文件，我已经修改好了。
        ########################开始训练#############################
        #training_loop_times=100  # 把进度条分为10分，所以训练也分解为 10次。
        # for i in range(training_loop_times):  #这个for，只是为了进度条的显示，所以分成 10次来训练。
        #     per_step_result=model.fit(x, y,validation_split=0.33, epochs=int(max(training_times/training_loop_times,1)), batch_size=training_batch_size,verbose = 1,shuffle=True) #这一行没有带callbacks，所以无法使用TensorBoard
        #     final_result_loss=str(per_step_result.history['loss'][0])[:5]
        #     final_result_acc=str(per_step_result.history['acc'][0])[:5]
        #     print("Training loop times:",i,"/100")
        #     ui.label.setText("开始机器学习你的脑纹。目前的损失率为"+final_result_loss+"  目前的准确率为"+final_result_acc)
        #     ui.progressBar.setProperty("value", (i+1)*100/training_loop_times)
        #     QApplication.processEvents()  #用于PyQt界面的刷新，保证流畅程度。
        early_stopping = EarlyStopping(monitor='val_loss',patience=int(training_times*0.2),verbose=1,mode='auto') 
        training_process_bar=Training_process_bar()
        per_training_step_result=model.fit(x, y, validation_split=0.33,epochs=training_times, batch_size=training_batch_size,verbose = 1,shuffle=True,callbacks=[training_process_bar]) #这一行带callbacks，是为了使用TensorBoard
        
        model.save(filename+'_key.h5')
        print("把训练好的神经网络保存在:",filename,'_key.h5')

        directly_load_model_flag= False
        ########################训练完毕，并保存训练好的神经网络#############################
        ui.label.setText("你的脑纹锁设置成功！")
        #ui.label.setText("你的脑纹锁设置成功！最终的损失率为"+final_result_loss+"  最终的准确率为"+final_result_acc)
        ui.label.repaint()
        ui.pushButton.setText("开始匹配测试")   
        ui.pushButton.repaint()
        ui.progressBar.setProperty("value",0)
        step=3
        ui.label_2.setText("目前载入的是"+filename+"的已经训练好的脑纹")
        ui.label_2.repaint()
        enable_Gui()

    if step==1:   #### 录制脑电波
        disable_Gui()
        ui.label.setText("开始录制你的脑纹信息，请保持不动。")
        ui.label.update()
        ui.pushButton.setText("1-录制脑纹中...")   
        ui.pushButton.repaint()
        s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.connect((server_address,5204))
        f = open(filename+'EEG.txt', 'w')
        for k in range(total_EEG_data_number):
            EEG_data=(s.recv(1024).decode('utf-8'))
            EEG_data_to_write=EEG_data+'\r\n'
            f.write(EEG_data_to_write)
            ui.progressBar.setProperty("value", k/total_EEG_data_number*100+5)
            QApplication.processEvents()  #用于PyQt界面的刷新，保证流畅程度。
        f.close()
        step=2
        ui.label_2.update()
        ui.label.setText("接下来神经网络开始学习并设置你的脑纹。")
        ui.label.update()
        ui.pushButton.setText("2-开始机器学习")   
        ui.pushButton.repaint()
        enable_Gui()
    if step==0:
        ui.label.setText("返回第一步，请输入你的名字。")
        ui.label.update()
        ui.pushButton.setText("1-重新开始录制脑纹")   
        ui.pushButton.repaint()
        ui.label_2.setText("目前没有载入任何已经训练好的脑纹。")
        ui.label_2.repaint()
        step=1

def reset():
    global step,directly_load_model_flag
    step=0
    directly_load_model_flag=False
    directly_load_EEG_flag=False
    ui.label_2.setText("目前没有载入任何已经训练好的脑纹。")
    ui.label_2.repaint()
    # ui.lineEdit.setText("请输入你的名字")
    ui.lineEdit.setText("请输入你的名字")
    action()  

def update_labels():
    ui.label_4.setText("EEG总采样帧数："+str(total_EEG_data_number))
    ui.label_5.setText("每次训练的帧组数："+str(training_batch_size))
    ui.label_6.setText("训练的总次数："+str(training_times))
    ui.label_7.setText("用于识别的帧组数："+str(total_EEG_group_number_for_test))
    ui.label_8.setText("每帧组的帧数："+str(time_steps))

def apply_parameters():
    global server_address,total_EEG_data_number,training_times,training_batch_size,total_EEG_group_number_for_test,time_steps
    server_address=ui.lineEdit_2.text()
    #total_EEG_data_number=int(max(ui.horizontalSlider.value()*total_EEG_data_number_times,100)) #读取n条EEG数据
    total_EEG_data_number=int(max((int(ui.horizontalSlider.value()*total_EEG_data_number_times/time_steps)+1)*time_steps,time_steps)) #读取n条EEG数据,必须是time_steps的整数。
    training_batch_size=int(max(ui.horizontalSlider_2.value()*2,1))  #每次训练输入的EEG帧数
    training_times=int(max(ui.horizontalSlider_3.value()*10,10)) #训练的次数
    total_EEG_group_number_for_test=int(max(ui.horizontalSlider_4.value()/5,1))
    time_steps=int(ui.horizontalSlider_5.value()*(time_steps/50))
    
    update_labels()




def reset_parameters():
    global server_address,total_EEG_data_number,training_times,training_batch_size,total_EEG_group_number_for_test,time_steps
    ui.horizontalSlider.setValue(50)
    ui.horizontalSlider_2.setValue(50)
    ui.horizontalSlider_3.setValue(50)
    ui.horizontalSlider_4.setValue(50)
    ui.horizontalSlider_5.setValue(50)
    

    ui.lineEdit_2.setText("127.0.0.1")
    server_address=ui.lineEdit_2.text()
    total_EEG_data_number=int(max((int(ui.horizontalSlider.value()*total_EEG_data_number_times/time_steps)+1)*time_steps,time_steps)) #读取n条EEG数据,必须是time_steps的整数。
    training_batch_size=int(max(ui.horizontalSlider_2.value()*2,1))  #每次训练输入的EEG帧数
    training_times=int(max(ui.horizontalSlider_3.value()*10,10)) #训练的次数
    total_EEG_group_number_for_test=int(max(ui.horizontalSlider_4.value()/5,1))
    time_steps=int(ui.horizontalSlider_5.value()*(time_steps/50))
    
    #print("reset",server_address,total_EEG_data_number,training_times,training_batch_size,total_EEG_group_number_for_test)
    update_labels()

def load_saved_EEG_for_training():
    global step,directly_load_model_flag,openfile_name,directly_load_filename,directly_load_EEG_flag
    openfile_name = QFileDialog.getOpenFileName(ui,'选择文件','','EEG files(*EEG.txt)')
    print(openfile_name[0])
    directly_load_filename=openfile_name[0]

    if directly_load_filename!='':
        directly_load_EEG_flag=True
        ui.label.setText("你预录制的脑纹已经载入成功，但是还未经过训练。")
        ui.label.repaint()
        ui.pushButton.setText("开始训练")   
        ui.pushButton.repaint()
        ui.progressBar.setProperty("value",0)
        step=2
        fileinfo = QFileInfo(directly_load_filename);
        fileName = fileinfo.fileName();
        ui.label_9.setText("目前载入的已经录制好的未经训练脑纹是: "+fileName[:-7])
        ui.label_9.repaint()
        ui.lineEdit.setText(fileName[:-7])
        ui.lineEdit.repaint()
        ui.label_2.setEnabled(False)
        ui.label_9.setEnabled(True)
    




def load_saved_model():
    global step,directly_load_model_flag,openfile_name,directly_load_filename
    openfile_name = QFileDialog.getOpenFileName(ui,'选择文件','','h5 files(*.h5)')
    print(openfile_name[0])
    directly_load_filename=openfile_name[0]

    if directly_load_filename!='':
        directly_load_model_flag=True
        ui.label.setText("你的脑纹锁设置成功！")
        ui.label.repaint()
        ui.pushButton.setText("开始匹配测试")   
        ui.pushButton.repaint()
        ui.progressBar.setProperty("value",0)
        step=3
        fileinfo = QFileInfo(directly_load_filename);
        fileName = fileinfo.fileName();
        ui.label_2.setText("目前载入的已经训练好的脑纹是: "+fileName[:-7])
        ui.label_2.repaint()
        ui.lineEdit.setText(fileName[:-7])
        ui.lineEdit.repaint()
        ui.label_2.setEnabled(True)
        ui.label_9.setEnabled(False)

def disable_Gui():
    ui.pushButton.setEnabled(False)
    ui.pushButton_2.setEnabled(False)
    ui.pushButton_3.setEnabled(False)
    ui.pushButton_4.setEnabled(False)
    ui.pushButton_5.setEnabled(False)
    ui.horizontalSlider.setEnabled(False)
    ui.horizontalSlider_2.setEnabled(False)
    ui.horizontalSlider_3.setEnabled(False)
    ui.horizontalSlider_4.setEnabled(False)
    ui.horizontalSlider_5.setEnabled(False)
    ui.lineEdit.setEnabled(False)
    ui.lineEdit_2.setEnabled(False)

def enable_Gui():
    ui.pushButton.setEnabled(True)
    ui.pushButton_2.setEnabled(True)
    ui.pushButton_3.setEnabled(True)
    ui.pushButton_4.setEnabled(True)
    ui.pushButton_5.setEnabled(True)
    ui.horizontalSlider.setEnabled(True)
    ui.horizontalSlider_2.setEnabled(True)
    ui.horizontalSlider_3.setEnabled(True)
    ui.horizontalSlider_4.setEnabled(True)
    ui.horizontalSlider_5.setEnabled(False)
    ui.lineEdit.setEnabled(True)
    ui.lineEdit_2.setEnabled(True)


#################下面是主程序##############################

app = QApplication(sys.argv)
mainWindow = QMainWindow()
ui =Ui_MainWindow()
ui.setupUi(mainWindow)

history = LossHistory()

# setup stylesheet
#app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())


ui.progressBar.setProperty("value", 0)
ui.pushButton.setText("1-开始录制脑纹")
ui.label_2.setText("目前没有载入任何已经训练好的脑纹。")
ui.label_2.repaint()   
reset_parameters()
ui.horizontalSlider_5.setEnabled(False)
ui.pushButton.clicked.connect(action)
ui.pushButton_2.clicked.connect(reset)
ui.pushButton_3.clicked.connect(load_saved_model)
ui.pushButton_4.clicked.connect(reset_parameters)
ui.pushButton_5.clicked.connect(load_saved_EEG_for_training)

ui.horizontalSlider.sliderMoved.connect(apply_parameters)
ui.horizontalSlider_2.sliderMoved.connect(apply_parameters)
ui.horizontalSlider_3.sliderMoved.connect(apply_parameters)
ui.horizontalSlider_4.sliderMoved.connect(apply_parameters)
ui.horizontalSlider_5.sliderMoved.connect(apply_parameters)

mainWindow.show()
sys.exit(app.exec_())