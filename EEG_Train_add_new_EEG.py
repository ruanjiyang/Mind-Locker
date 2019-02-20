#!!!!!
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import numpy as np
####################### 从 EEG_data_A.txt 和  EEG_data_B.txt 读取 EEG数据############################

total_EEG_data_number=1000#读取n条EEG数据
total_EEG_Features=24  #这是固定的。每一条EEG数据都有24个参数值。
training_times=500 #训练的次数
learn_rate=0.02 #学习率


torch.set_default_dtype(torch.float64)


f = open('EEG record\EEG_data_Ruanjiyang.txt', 'r')
All_EEG_data_lines=f.readlines()
EEG_data_A=torch.Tensor(total_EEG_data_number,total_EEG_Features)

for k in range(total_EEG_data_number):
    EEG_data_one_line=(All_EEG_data_lines[k].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份
    # print(EEG_data_one_line)
    for i in range(total_EEG_Features):
        EEG_data_A[k][i]=float(EEG_data_one_line[i])
f.close()
###############读取EEG_data_A.txt完毕######################

f = open('EEG record\EEG_data_Ruanjingheng.txt', 'r')
All_EEG_data_lines=f.readlines()
EEG_data_B=torch.Tensor(total_EEG_data_number,total_EEG_Features)

for k in range(total_EEG_data_number):
    EEG_data_one_line=(All_EEG_data_lines[k].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份

    for i in range(total_EEG_Features):
        EEG_data_B[k][i]=float(EEG_data_one_line[i])
f.close()
###############读取EEG_data_B.txt完毕######################

###########################从 OPENBCI读取 EEG数据结束##########################

########################开始生成 数据样本
# n_data = torch.ones(total_EEG_data_number, total_EEG_Features)   
# x0 = torch.normal(EEG_data_A, 20)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.ones(total_EEG_data_number)               # class0 y data (tensor), shape=(100, 1)
# x1 = torch.normal(EEG_data_B, 20)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.zeros(total_EEG_data_number)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((EEG_data_A,EEG_data_B), ).type(torch.DoubleTensor)  # shape (200, 2) FloatTensor = 32-bit floating
# print(x.size())
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer




################ 数据整理!!!! ###################
for i in range(total_EEG_data_number*2):
     x[i][2]=x[i][2]/10
     x[i][3]=x[i][3]/10
     x[i][4]=x[i][4]/10
     x[i][20]=x[i][4]/2
     x[i][21]=x[i][4]/2

################数据整理###################

####################### 绘制原始数据 ##########################

###########准备X轴##################
x_zhou=torch.ones(x.size())
for i in range(total_EEG_data_number*2):
    for j in range(total_EEG_Features):
        x_zhou[i][j]=j+random.randint(-4,4)/10  #加了一些随机量，否则都在一直线上了。
# ####################################
# tensor_2_draw=x  ##指定绘制哪个Tensor
# prediction = torch.max(tensor_2_draw, 1)[1]  
# pred_y = prediction.data.numpy()
# for i in range(0,23,1):
#     plt.scatter(x_zhou.data.numpy()[:, i], tensor_2_draw.data.numpy()[:, i], c=y.data.numpy(), s=5+ tensor_2_draw.data.numpy()[:, i]*4, lw=1, cmap='RdYlGn',alpha=0.2)
# plt.show()
###############################################################

###################转入到 Cuda#####################3
if torch.cuda.is_available():
    print("Cuda is installed")
    x_zhou=x_zhou.cuda()
    x=x.cuda()
    y=y.cuda()
    print("Change to Cuda")

#################### 开始训练#####################################

net=torch.load('ruanjiyang_net.pkl')  #读取训练好的结果

optimizer = torch.optim.SGD(net.parameters(), lr=learn_rate)
loss_func = torch.nn.CrossEntropyLoss().cuda()  # the target label is NOT an one-hotted
#loss_func = torch.nn.MSELoss()


plt.ion()   # something about plotting

for t in range(training_times):
    out = net(x)                 # input x and predict based on x
    loss = loss_func(out, y)    # must be (1. nn output, 2. target), the target label is NOT one-hotted
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.cpu().numpy()
        target_y = y.data.cpu().numpy()

        #plt.plot(x_zhou.data.numpy(), x.data.numpy(), color=pred_y, linewidth=1, alpha=0.6)
        for i in range(0,24,1):
            plt.scatter(x_zhou.data.cpu().numpy()[:, i], x.data.cpu().numpy()[:, i], c=pred_y, s=0.2+x.data.cpu().numpy()[:, i]*5, lw=1, cmap='RdYlGn',alpha=0.2)
        loss_output=loss*100
        accuracy = (float((pred_y == target_y).astype(int).sum()) / float(target_y.size))*100
        out_text='Accuracy=%.2f'%accuracy +'%    '+ 'Loss=%.2f' % loss_output +'%\n'+ 'Learning times=%d   ' %t  + 'Learning Rate=%.3f' %learn_rate
        plt.text(5,np.max(x.data.cpu().numpy())-5, out_text,fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

# #################### 训练结束 #####################################


torch.save(net, 'ruanjiyang_net.pkl')  #保存好训练结果

