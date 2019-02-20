

################## 项目名称： “Mind Locker：一种基于神经网络与深度学习的脑纹锁系统” ####################

################## 拟解决的关键问题 #####################
'''
我们所使用的主流生物特征加密方法，如指纹/眼球加密/人脸识别，因为其特征值数量有限/重复率高等因素，几乎一一被攻克，
安全性受到极大挑战。比如，目前英伟达的GAN人工智能生成的虚拟人脸技术，就攻陷了目前几乎所有的人脸识别系统。所以，在
一些极重要的应用场合，传统的生物特征加密方法的安全性，已经受到了严重的挑战。除了人脸，眼球，指纹外，我们还有其
他更为先进与安全的生物特征加密手段么？ 答案是：人的大脑。每个人的大脑，其生物特征都是真正意义上的独一无二。用大脑的
生物特征作为一种加密手段，这就是我们今天所要设计的一种基于神经网络与深度学习的脑纹锁，我把它命名为“Mind Locker".

'''
################ 主要创意点 革新点 与建议 ################
'''
大脑生物特征复杂多变，如何在海量的EEG数据中提取可供识别的生物特征，从而构建起一个加密系统？
举例说，将A,B,C 三人各自一千条脑电波EEG数据信息，完全打乱，再混合在一起。用传统的数学方法，我们几乎不可能再
将它们一一区分出来。因为每一条EEG均蕴含了24条生物特征信息，它们相互之间的非线性数学关系一起构成了这颗大脑的完整生
物特征网。所以设计的关键在于，如何提取这24条生物特征信息之间的点对点的内在蕴含关系。如需破解，则必须完全再现
所有点对点网元间的数学关系，所以几乎不存在破解的可能性。

借助目前神经网络和深度学习技术，我们实现了让计算机自己去学习与提取上述的生物特征值之间的关系，达到了99.9%以上的 
的识别率（采用的训练模型是2000条EEG信息，神经元结构为[24*100*100*2]，训练次数为5000次）,从而完成了脑纹
锁系统最核心部分的算法设计。

'''

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


f = open('EEG_data_Ruanjiyang.txt', 'r')
All_EEG_data_lines=f.readlines()
EEG_data_A=torch.Tensor(total_EEG_data_number,total_EEG_Features)

for k in range(total_EEG_data_number):
    EEG_data_one_line=(All_EEG_data_lines[k].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份
    # print(EEG_data_one_line)
    for i in range(total_EEG_Features):
        EEG_data_A[k][i]=float(EEG_data_one_line[i])
f.close()
###############读取EEG_data_A.txt完毕######################

f = open('EEG_data_Ruanminli.txt', 'r')
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
n_data = torch.ones(total_EEG_data_number, total_EEG_Features)   
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

net = torch.nn.Sequential(
    torch.nn.Linear(total_EEG_Features,100).cuda(),
    torch.nn.ReLU().cuda(),
    torch.nn.Linear(100, 100).cuda(),
    torch.nn.ReLU().cuda(),
    torch.nn.Linear(100, 2).cuda()
)

if torch.cuda.is_available():
    net=net.cuda()

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


# ########################下面开始测试阶段############################

# f = open('EEG_data_Ruan_2.txt', 'r')
# All_EEG_data_lines=f.readlines()
# load_EEG_data=torch.Tensor(total_EEG_data_number,total_EEG_Features).cuda()

# for k in range(total_EEG_data_number):
#     EEG_data_one_line=(All_EEG_data_lines[k].split('A'))  ####按照字符"A"来截断每一条EEG数据，分割成24小份

#     for i in range(total_EEG_Features):
#         load_EEG_data[k][i]=float(EEG_data_one_line[i])

# f.close()

# score_false=0 # false 的总得分 
# score_true=0 # true的总得分 
# correct_counter=0  # 单个正确波的总数量。
# result = net(load_EEG_data)
# # print(result)
# for i in range(total_EEG_data_number):
#     score_false=score_false+result[i][0]
#     score_true=score_true+ result[i][1]
#     if result[i][0]<0 and result[i][1]>0:
#         correct_counter=correct_counter+1
#         print("Corret")
#     else:
#         print("Wrong!!!")
# if score_false<0 and score_true>0:
#     print("Yes, it is your EEG!")
#     print("score_false=%d"%score_false+"   score_true=%d"%score_true)
# else:
#     print("No, it is not your EEG!")
#     print("score_false=%d"%score_false+"   score_true=%d"%score_true)
# print("Total correct rate=%10.3f"%(correct_counter/total_EEG_data_number*100))
# # result = net(EEG_data_B)
# print(result)
