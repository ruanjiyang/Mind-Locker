import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import numpy as np


net=torch.load('ruanjiyang_net.pkl')  #读取训练好的结果


total_EEG_data_number=1000#读取n条EEG数据
total_EEG_Features=24  #这是固定的。每一条EEG数据都有24个参数值。
training_times=500 #训练的次数
learn_rate=0.02 #学习率


torch.set_default_dtype(torch.float64)

########################下面开始测试阶段############################

# f = open('EEG_data_Ruan_2.txt', 'r')  #读取需要测试的EEG数据
f = open('EEG_data_Ruanjingheng.txt', 'r')  #读取需要测试的EEG数据
All_EEG_data_lines=f.readlines()
load_EEG_data=torch.Tensor(total_EEG_data_number,total_EEG_Features).cuda()

for k in range(total_EEG_data_number):
    EEG_data_one_line=(All_EEG_data_lines[k].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份

    for i in range(total_EEG_Features):
        load_EEG_data[k][i]=float(EEG_data_one_line[i])

f.close()

score_false=0 # false 的总得分 
score_true=0 # true的总得分 
correct_counter=0  # 单个正确波的总数量。
result = net(load_EEG_data)
# print(result)
for i in range(total_EEG_data_number):
    score_false=score_false+result[i][0]
    score_true=score_true+ result[i][1]
    if result[i][0]<0 and result[i][1]>0:
        correct_counter=correct_counter+1
        # print("Corret")
    # else:
        # print("Wrong!!!")
if score_false<0 and score_true>0:
    print("Yes, it is your EEG!")
else:
    print("No, it is not your EEG!")
print("score_false=%d"%score_false+"   score_true=%d"%score_true)
print("Total correct rate=%10.3f"%(correct_counter/total_EEG_data_number*100)+'%')
# result = net(EEG_data_B)
# print(result)