import numpy as np
import numpy as np
import random

####################  全局超参数 ##############################
total_EEG_data_number=2000 #读取n条EEG数据 注意这个数字必须是time_steps 100的倍数
total_EEG_Features=18  #这是固定的。每一条EEG数据都有24个参数值。
training_times=1000 #训练的次数
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


 
# # X shape (60,000 28x28), y shape (10,000, )
# (x_train, _), (x_test, y_test) = mnist.load_data()
 
# # 数据预处理
# x_train = x_train.astype('float32') / 255. - 0.5       # minmax_normalized
# x_test = x_test.astype('float32') / 255. - 0.5         # minmax_normalized
# x_train = x_train.reshape((x_train.shape[0], -1))
# x_test = x_test.reshape((x_test.shape[0], -1))
# print(x_train.shape)
# print(x_test.shape)
 

f = open('2_EEG.txt', 'r') #这是用于参与训练的他人EEG
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
print(type(EEG_data_B[0][0]))
print('max=',EEG_data_B.max())