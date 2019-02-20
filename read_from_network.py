import socket 
import torch

import socket 

total_EEG_data_number=1000 #读取n条EEG数据
total_EEG_Features=24  #这是固定的。每一条EEG数据都有24个参数值。
filename='EEG_data_Ruanjingheng.txt' #写入文件名

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect(('127.0.0.1',5204))
f = open(filename, 'w')
for k in range(total_EEG_data_number):
  EEG_data=(s.recv(1024).decode('utf-8'))
  f.write(EEG_data)
  f.write('\n')
f.close()
