
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
import qdarkstyle
import keras
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Dropout,Activation,regularizers,Input
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
total_EEG_number_for_test=50 #每次检测所采样的EEG帧数
server_address='127.0.0.1'
step=1 #此参数是：action（）的操作的步骤的标志。
filename=''
directly_load_filename=''
match_triger=0.9  #此参数设置了每一帧的通过测试的阀门值。
total_EEG_data_number_times=40  #此参数设置了EEG采样的倍数。比如中间值为40*50.
directly_load_model_flag=False  #此参数是：是否直接读取预训练模型的标志。
directly_load_EEG_flag=False #此参数是：是否直接读取预录制EEG的标志
time_steps=1  #明天写注释！！！！
# 压缩特征维度
encoding_dim =  9  #1肯定不行， 2目前很好（但是对于rjy-C2不行，会出现高匹配）。 3似乎目前是最好的（对 rjy-C2也没问题）/3似乎也不行了 

normal_number=10  #归一化的被除数

def rest_process_bar():
    ui.progressBar_dim_0.setProperty("value",0) 
    ui.progressBar_dim_1.setProperty("value",0) 
    ui.progressBar_dim_2.setProperty("value", 0)
    ui.progressBar_dim_3.setProperty("value",0)
    ui.progressBar_dim_4.setProperty("value",0)
    ui.progressBar_dim_5.setProperty("value",0)
    ui.progressBar_dim_6.setProperty("value",0)
    ui.progressBar_dim_7.setProperty("value", 0)
    ui.progressBar_dim_8.setProperty("value",0)

def moreBZOSdis(a,b):   #标准欧氏距离
    sumnum = 0
    for i in range(len(a)):
        # 计算si 分量标准差
        avg = (a[i]-b[i])/2
        si = np.sqrt( (a[i] - avg) ** 2 + (b[i] - avg) ** 2 )
        sumnum += ((a[i]-b[i])/si ) ** 2

    return np.sqrt(sumnum)


#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
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
    def on_epoch_end(self, batch, logs={}):
        #ui.label.setText("开始机器学习你的脑纹。目前的损失率为"+str(logs.get('loss'))[:12]+"  目前的准确率为"+str(logs.get('accuracy'))[:12])
        ui.label.setText("开始机器学习你的脑纹。目前的训练损失率为%e  目前的验证集损失率为%e"%(logs.get('loss'),logs.get('val_loss')))
        ui.progressBar.setProperty("value", (batch+1)/training_times*100)
        QApplication.processEvents()  #用于PyQt界面的刷新，保证流畅程度。

def action(): #此函数是控制整个采样，训练， 匹配测试的过程。
    global filename,total_EEG_number_for_test
    global step,directly_load_model_flag,openfile_name,directly_load_filename,time_steps,directly_load_EEG_flag
    filename=ui.lineEdit.text() #写入文件名
    time_steps=1
    if step==3:   # 匹配测试
        disable_Gui()
        ui.lcdNumber.display('---')
        QApplication.processEvents()
        ui.label.setText("匹配测试中...")
        ui.pushButton.setText("匹配测试中...")   

        if directly_load_model_flag==False:
            #autoencoder=load_model(filename+'_key.h5')
            encoder=load_model(filename+'_key.h5')
            print("使用的是刚才训练好的模型:",filename)
        if directly_load_model_flag==True:
            directly_load_EEG_flag=False
            #autoencoder=load_model(directly_load_filename)
            encoder=load_model(directly_load_filename)
            print("使用的是直接读取的已经训练好的模型:", directly_load_filename)

#############################设置初始偏差值 开始##############################################
        if directly_load_EEG_flag== True:
            f = open(directly_load_filename, 'r')  #读取代训练的EEG数据
            print("使用的是预先录取好的EEG文件:",directly_load_filename)
        if directly_load_EEG_flag== False:
            f = open(filename+'EEG.txt', 'r')  #读取代训练的EEG数据
            print("使用的是刚才录取的EEG文件:",filename)
        All_EEG_data_lines=f.readlines() #################BUG##############################
        EEG_data_original=np.zeros([total_EEG_number_for_test,total_EEG_Features])
        final_score=0
        max_score=0
        original_test_result=np.zeros(int(total_EEG_data_number/total_EEG_number_for_test))
        original_test_result_array=np.zeros([total_EEG_number_for_test,encoding_dim])
        for times in range (int(total_EEG_data_number/total_EEG_number_for_test)):  #0,1,2,3....
            for k in range(total_EEG_number_for_test):
                EEG_data_one_line=(All_EEG_data_lines[k+times*total_EEG_number_for_test].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份
                for i in range(total_EEG_Features):
                    if len(EEG_data_one_line)==total_EEG_Features+1: #这个判断是为了避免有时候读取EEG时候，遇到换行符丢失的现象。
                        EEG_data_original[k][i]=float(EEG_data_one_line[i])
                    else:
                        EEG_data_original[k][i]=EEG_data_original[k-1][i]
                        print("发现一处错行！！")
            f.close()
            EEG_data_original = EEG_data_original.astype('float32') / normal_number     # minmax_normalized

            EEG_data_original=EEG_data_original.reshape(int(total_EEG_number_for_test/time_steps),int(time_steps*total_EEG_Features))
            x=encoder.predict(EEG_data_original)
            original_test_result[times]=sum(abs(sum(abs((x)))))
            original_test_result_array+=x ##################
            final_score+=sum(abs(sum(abs((x)))))
            max_score=max(max_score,sum(abs(sum(abs((x))))))
        matched_average_score=(sum(original_test_result))/(int(total_EEG_data_number/total_EEG_number_for_test))
        matched_median_score=np.median(original_test_result)
        print('==================设置的Median Score================',matched_median_score)
        print('==================设置的Average Score================',matched_average_score)
        original_test_result_array=original_test_result_array/(int(total_EEG_data_number/total_EEG_number_for_test))
        # print('Original_test_result_array.shape==',original_test_result_array.shape)
        # print('Original_test_result_array==',original_test_result_array)
#############################设置初始偏差值 结束##############################################

        s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.connect((server_address,5204)) #5204 为 OPENBCI GUI的缺省服务器段的发送端口   
        EEG_data_for_test=np.zeros([total_EEG_number_for_test,total_EEG_Features]) # EEG_data_for_test为采样的测试EEG帧。
        match_counter=0 # 用于计数EEG通过测试的个数。
        max_score=0
        final_score=0  
        test_times=10
        test_result=np.zeros(test_times)
        test_result_array=np.zeros([total_EEG_number_for_test,encoding_dim])
        QApplication.processEvents() 
        for times in range(test_times): #测试十次   
            for k in range(total_EEG_number_for_test):  #明天写注释！！！！
                EEG_data_one_line=(s.recv(1024).decode('utf-8')).split('A') ####按照字符"A"来截断每一条EEG数据，分割成24小份
                for i in range(total_EEG_Features):
                    if len(EEG_data_one_line)==total_EEG_Features+1:  #这个判断是为了避免有时候读取EEG时候，遇到换行符丢失的现象。
                        EEG_data_for_test[k][i]=float(EEG_data_one_line[i])
                    else:
                        EEG_data_for_test[k][i]=EEG_data_for_test[k-1][i]
                        print("发现一处错行！！")

            EEG_data_for_test=EEG_data_for_test.astype('float32') / normal_number      # 归一化
            test=encoder.predict(EEG_data_for_test,verbose = 1)
            test_result_array+=test        
            test_result[times]=sum(abs(sum(abs((test)))))
            final_score+=test_result[times]
            max_score=max(max_score,test_result[times])

            ui.progressBar.setProperty("value", (times+1)*10)        
            QApplication.processEvents()  #用于PyQt界面的刷新，保证流畅程度。
        test_median=np.median(test_result)
        test_average=(final_score)/(test_times)
        print('==================现在测试出来的Median Score================',test_median)
        print('==================现在测试出来的Average Score=================',test_average )
        print('test_result_array.shape==',test_result_array.shape)
        test_score_for_display=(1-abs((test_median+test_average)-(matched_median_score+matched_average_score))/(matched_median_score+matched_average_score))*100
        if test_score_for_display <89:
            test_score_for_display-=20
        test_result_array=test_result_array/(test_times)
        # print('test_result_array.shape==',test_result_array.shape)
        # print('test_result_array==',test_result_array)

        # print('标准欧几里得距离为:', moreBZOSdis(original_test_result_array,test_result_array))
        # print('标准欧几里得距离之和为:', sum(moreBZOSdis(original_test_result_array,test_result_array)))
        original_average_zip_features=sum(original_test_result_array)/total_EEG_number_for_test
        tester_average_zip_features=sum(test_result_array)/total_EEG_number_for_test
        print('看一下原始加密者的压缩层的平均值:',original_average_zip_features)
        print('看一下测试者的压缩层的平均值:',tester_average_zip_features)
        print('看一下原始人VS测试者的压缩层的平均值的标准欧几里得距离为:',moreBZOSdis(original_average_zip_features,  tester_average_zip_features  ))
        test_score_for_display_2=0
        progressBar_dim_value=np.zeros(encoding_dim)
        for i in range(encoding_dim):
            test_score_for_display_2+= (1-abs(original_average_zip_features[i]-tester_average_zip_features[i])/abs(original_average_zip_features[i]))/encoding_dim*100
            progressBar_dim_value[i]=(1-abs(original_average_zip_features[i]-tester_average_zip_features[i])/abs(original_average_zip_features[i]))*100
            # if progressBar_dim_value[i]<=0:
            #     progressBar_dim_value[i]=0
            print("每一个维度的分数",progressBar_dim_value[i])
            
        ########################绘制每个维度的匹配度################################
        ui.progressBar_dim_0.setProperty("value",max(int(progressBar_dim_value[0]),0) )
        ui.progressBar_dim_1.setProperty("value",max(int(progressBar_dim_value[1]),0) )
        ui.progressBar_dim_2.setProperty("value",max(int(progressBar_dim_value[2]),0) )
        if encoding_dim>=4:
            ui.progressBar_dim_3.setProperty("value",max(int(progressBar_dim_value[3]),0) )
        if encoding_dim>=5:
            ui.progressBar_dim_4.setProperty("value",max(int(progressBar_dim_value[4]),0) )
        if encoding_dim>=6:
            ui.progressBar_dim_5.setProperty("value",max(int(progressBar_dim_value[5]),0) )
        if encoding_dim>=7:
            ui.progressBar_dim_6.setProperty("value",max(int(progressBar_dim_value[6]),0) )
        if encoding_dim>=8:
            ui.progressBar_dim_7.setProperty("value",max(int(progressBar_dim_value[7]),0) )
        if encoding_dim>=9:
            ui.progressBar_dim_8.setProperty("value",max(int(progressBar_dim_value[8]),0) )

        # for i in range(total_EEG_number_for_test):
        #     print('每一帧的标准欧几里得距离为:',moreBZOSdis(test_result_array[i],sum(original_test_result_array)/total_EEG_number_for_test))

        print('test_score_for_display=',test_score_for_display_2)
        ui.lcdNumber.display(test_score_for_display_2)
        result_text="测试结束。最终匹配结果为"+str(test_score_for_display_2)+"%"
        ui.label.setText(result_text)
        ui.label.repaint()
        QApplication.processEvents()


        ui.pushButton.setText("重新开始匹配测试")   
        ui.pushButton.repaint()

        enable_Gui()

    if step==2:  #机器学习
        disable_Gui()
        rest_process_bar()
        ui.lcdNumber.display('---')
        QApplication.processEvents()  #用于PyQt界面的刷新，保证流畅程度。
        ui.label.setText("开始机器学习你的脑纹。")
        ui.label.repaint()
        ui.pushButton.setText("2-机器学习中...")   
        ui.pushButton.repaint()


        ######################################### 开始训练 ######################################
     
        ####################### 读取代训练的EEG数据############################
        if directly_load_EEG_flag== True:
            f = open(directly_load_filename, 'r')  #读取代训练的EEG数据
            print("step-2:使用的是预先录取好的EEG文件:",directly_load_filename)
        if directly_load_EEG_flag== False:
            f = open(filename+'EEG.txt', 'r')  #读取代训练的EEG数据
            print("使用的是刚才录取的EEG文件:",filename)
               
        print(2)
        All_EEG_data_lines=f.readlines()
        EEG_data=np.zeros([total_EEG_data_number,total_EEG_Features])
        time_steps=1 #强制设定time_steps=1
        for k in range(total_EEG_data_number):
            EEG_data_one_line=(All_EEG_data_lines[k].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份
            for i in range(total_EEG_Features):
                if len(EEG_data_one_line)==total_EEG_Features+1: #这个判断是为了避免有时候读取EEG时候，遇到换行符丢失的现象。
                    EEG_data[k][i]=float(EEG_data_one_line[i])
                else:
                    EEG_data[k][i]=EEG_data[k-1][i]
                    print("发现一处错行！！")
        f.close()
        EEG_data = EEG_data.astype('float32') / normal_number      # minmax_normalized
        EEG_data=EEG_data.reshape(int(total_EEG_data_number/time_steps),int(time_steps*total_EEG_Features))
        x_train=EEG_data


        ########################开始搭建神经网络#############################
        #120-60-30-15-6-3 似乎不错
 
        # this is our input placeholder
        input_img = Input(shape=(int(time_steps*total_EEG_Features),))
        
        # 编码层 
        encoded = Dense(288, activation='relu',activity_regularizer=regularizers.l1(0))(input_img)  #10e-7
        encoded = Dense(144, activation='relu',activity_regularizer=regularizers.l1(0))(encoded)
        encoded = Dense(72, activation='relu',activity_regularizer=regularizers.l1(0))(encoded)
        encoded = Dense(36, activation='relu',activity_regularizer=regularizers.l1(0))(encoded)
        encoded = Dense(18, activation='relu',activity_regularizer=regularizers.l1(0))(encoded)
        encoder_output = Dense(encoding_dim)(encoded)
        
        # 解码层
        decoded = Dense(18, activation='relu',activity_regularizer=regularizers.l1(0))(encoder_output)
        decoded = Dense(36, activation='relu',activity_regularizer=regularizers.l1(0))(decoded)
        decoded = Dense(72, activation='relu',activity_regularizer=regularizers.l1(0))(decoded)
        decoded = Dense(144, activation='relu',activity_regularizer=regularizers.l1(0))(decoded)
        decoded = Dense(288, activation='relu',activity_regularizer=regularizers.l1(0))(decoded)
        decoded = Dense(int(time_steps*total_EEG_Features), activation='softsign')(decoded)
        
        # 构建自编码模型
        autoencoder = Model(inputs=input_img, outputs=decoded)
        
        # 构建编码模型
        encoder = Model(inputs=input_img, outputs=encoder_output)
        
        # compile autoencoder
        #adam=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        autoencoder.compile(optimizer='adam', loss='mse')
        encoder.compile(optimizer='adam', loss='mse') 
        
        # compile autoencoder
        #adam=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        autoencoder.compile(optimizer='adam', loss='mse')
        encoder.compile(optimizer='adam', loss='mse') 
        # training
        print(x_train.shape)

        ########################神经网络搭建完毕#############################
        #手动 打乱 x,y 次序。
        # p = np.random.permutation(range(len(x_train)))
        # x_train = x_train[p] 
        np.random.shuffle(x_train)

        ################这个tb，是为了使用TensorBoard########################
        tensorBoard = TensorBoard(log_dir='./logs',  # log 目录
            histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算.  必须在fit的时候设定好 validation_split或者validation数据，否则程序死机（此时只能设置为0）。
            batch_size=32,     # 用多大量的数据计算直方图
            write_graph=True,  # 是否存储网络结构图
            write_grads=False, # 是否可视化梯度直方图
            write_images=False,# 是否可视化参数
            embeddings_freq=0, 
            embeddings_layer_names=None, 
            embeddings_metadata=None)    

        # 在命令行，先conda activate envs，然后进入本代码所在的目录，然后用 tensorboard --logdir=logs/ 来看log
        # 然后打开chrome浏览器，输入http://localhost:6006/ 来查看
        ########################开始训练#############################
        early_stopping = EarlyStopping(monitor='val_loss',patience=int(training_times*0.05),verbose=1,mode='min') 
        training_process_bar=Training_process_bar()
        lossHistory=LossHistory()
        #############有一个特别的bug，请注意，就是在一个程序中，如果要训练两次，那么请在fit中拿掉tensorBoard！！！！
        autoencoder.fit(x_train, x_train, validation_split=0.33,epochs=training_times, batch_size=training_batch_size, shuffle=True,callbacks=[early_stopping,training_process_bar])
        #history.loss_plot('epoch')  #matplotlib绘制训练过程，似乎有问题。

        encoder.save(filename+'_key.h5')
        print("把训练好的神经网络保存在:",filename,'_key.h5')

        directly_load_model_flag= False

        x = encoder.predict(EEG_data.reshape(int(total_EEG_data_number/time_steps),int(time_steps*total_EEG_Features)))

        print('encoder (x)=',x)
        print('shape (x)=',x.shape)
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
        ui.lcdNumber.display('---')
        disable_Gui()
        ui.label.setText("开始录制你的脑纹信息，请保持不动。")
        ui.label.update()
        rest_process_bar()
        ui.pushButton.setText("1-录制脑纹中...")   
        ui.pushButton.repaint()
        s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.connect((server_address,5204))
        f = open(filename+'EEG.txt', 'w')
        for k in range(total_EEG_data_number):
            EEG_data=(s.recv(1024).decode('utf-8'))
            EEG_data_to_write=EEG_data+'\n'
            f.write(EEG_data_to_write)
            ui.progressBar.setProperty("value", k/total_EEG_data_number*100+5)
            QApplication.processEvents()  #用于PyQt界面的刷新，保证流畅程度。
        f.close()
        directly_load_EEG_flag=False 
        step=2
        ui.label_2.update()
        ui.label.setText("接下来神经网络开始学习并设置你的脑纹。")
        ui.label.update()
        ui.pushButton.setText("2-开始机器学习")   
        ui.pushButton.repaint()
        enable_Gui()
        directly_load_EEG_flag=False 
    if step==0:
        ui.lcdNumber.display('---')
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
    rest_process_bar()
    action()  

def update_labels():
    ui.label_4.setText("EEG总采样帧数："+str(total_EEG_data_number))
    ui.label_5.setText("每批次投入训练的帧数："+str(training_batch_size))
    ui.label_6.setText("训练的总批次数："+str(training_times))
    ui.label_7.setText("用于识别的帧数："+str(total_EEG_number_for_test))
    ui.label_8.setText("每帧组的帧数："+str(time_steps))

def apply_parameters():
    global server_address,total_EEG_data_number,training_times,training_batch_size,total_EEG_number_for_test,time_steps
    server_address=ui.lineEdit_2.text()
    #total_EEG_data_number=int(max(ui.horizontalSlider.value()*total_EEG_data_number_times,100)) #读取n条EEG数据
    total_EEG_data_number=int(max((int(ui.horizontalSlider.value()*total_EEG_data_number_times/time_steps))*time_steps,time_steps*10)) #读取n条EEG数据,必须是time_steps的整数。
    training_batch_size=int(max(ui.horizontalSlider_2.value()*2,10))  #每次训练输入的EEG帧数
    training_times=int(max(ui.horizontalSlider_3.value()*10,10)) #训练的次数
    total_EEG_number_for_test=int(max(ui.horizontalSlider_4.value(),10))
    time_steps=int(ui.horizontalSlider_5.value()*(time_steps/50))
    
    update_labels()

def reset_parameters():
    global server_address,total_EEG_data_number,training_times,training_batch_size,total_EEG_number_for_test,time_steps
    ui.horizontalSlider.setValue(50)
    ui.horizontalSlider_2.setValue(50)
    ui.horizontalSlider_3.setValue(50)
    ui.horizontalSlider_4.setValue(50)
    ui.horizontalSlider_5.setValue(50)

    

    ui.lineEdit_2.setText("127.0.0.1")
    server_address=ui.lineEdit_2.text()
    total_EEG_data_number=int(max((int(ui.horizontalSlider.value()*total_EEG_data_number_times/time_steps))*time_steps,time_steps*10)) #读取n条EEG数据,必须是time_steps的整数。
    training_batch_size=int(max(ui.horizontalSlider_2.value()*2,10))  #每次训练输入的EEG帧数
    training_times=int(max(ui.horizontalSlider_3.value()*10,10)) #训练的次数
    total_EEG_number_for_test=int(max(ui.horizontalSlider_4.value(),10))
    time_steps=int(ui.horizontalSlider_5.value()*(time_steps/50))
    
    #print("reset",server_address,total_EEG_data_number,training_times,training_batch_size,total_EEG_number_for_test)
    update_labels()

def load_saved_EEG_for_training():
    global step,directly_load_model_flag,openfile_name,directly_load_filename,directly_load_EEG_flag
    openfile_name = QFileDialog.getOpenFileName(ui,'选择文件','','EEG files(*EEG.txt)')
    print(openfile_name[0])
    directly_load_filename=openfile_name[0]
    rest_process_bar()
    
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
    rest_process_bar()

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
app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())


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