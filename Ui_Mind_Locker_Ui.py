# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\PycharmProjects\Mind Locker\Mind_Locker_Ui.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1280, 937)
        MainWindow.setMaximumSize(QtCore.QSize(1280, 1024))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(70, 370, 271, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(70, 202, 271, 61))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(70, 450, 271, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(410, 540, 631, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(70, 70, 1201, 91))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.lcdNumber = QtWidgets.QLCDNumber(self.centralwidget)
        self.lcdNumber.setGeometry(QtCore.QRect(410, 200, 331, 291))
        font = QtGui.QFont()
        font.setPointSize(36)
        self.lcdNumber.setFont(font)
        self.lcdNumber.setLineWidth(2)
        self.lcdNumber.setMidLineWidth(0)
        self.lcdNumber.setDigitCount(3)
        self.lcdNumber.setObjectName("lcdNumber")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(380, 200, 20, 301))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(750, 200, 20, 301))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(70, 510, 1081, 21))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setGeometry(QtCore.QRect(70, 160, 1081, 21))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.label_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_1.setGeometry(QtCore.QRect(400, 20, 751, 61))
        font = QtGui.QFont()
        font.setPointSize(22)
        self.label_1.setFont(font)
        self.label_1.setObjectName("label_1")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(780, 450, 371, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(900, 190, 291, 251))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.label_5 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.verticalLayout.addWidget(self.label_5)
        self.label_6 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)
        self.label_7 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.verticalLayout.addWidget(self.label_7)
        self.label_8 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.verticalLayout.addWidget(self.label_8)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(780, 189, 100, 261))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.layoutWidget1)
        self.lineEdit_2.setFrame(True)
        self.lineEdit_2.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.verticalLayout_2.addWidget(self.lineEdit_2)
        self.horizontalSlider = QtWidgets.QSlider(self.layoutWidget1)
        self.horizontalSlider.setToolTip("")
        self.horizontalSlider.setProperty("value", 50)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.verticalLayout_2.addWidget(self.horizontalSlider)
        self.horizontalSlider_2 = QtWidgets.QSlider(self.layoutWidget1)
        self.horizontalSlider_2.setProperty("value", 50)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.verticalLayout_2.addWidget(self.horizontalSlider_2)
        self.horizontalSlider_3 = QtWidgets.QSlider(self.layoutWidget1)
        self.horizontalSlider_3.setProperty("value", 50)
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        self.verticalLayout_2.addWidget(self.horizontalSlider_3)
        self.horizontalSlider_4 = QtWidgets.QSlider(self.layoutWidget1)
        self.horizontalSlider_4.setProperty("value", 50)
        self.horizontalSlider_4.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_4.setObjectName("horizontalSlider_4")
        self.verticalLayout_2.addWidget(self.horizontalSlider_4)
        self.horizontalSlider_5 = QtWidgets.QSlider(self.layoutWidget1)
        self.horizontalSlider_5.setProperty("value", 50)
        self.horizontalSlider_5.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_5.setObjectName("horizontalSlider_5")
        self.verticalLayout_2.addWidget(self.horizontalSlider_5)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(70, 290, 271, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.lineEdit.setFont(font)
        self.lineEdit.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(70, 540, 271, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(70, 600, 271, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(410, 600, 631, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_9.setFont(font)
        self.label_9.setText("")
        self.label_9.setObjectName("label_9")
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setGeometry(QtCore.QRect(70, 660, 1081, 21))
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.progressBar_dim_0 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_dim_0.setGeometry(QtCore.QRect(230, 710, 311, 23))
        self.progressBar_dim_0.setProperty("value", 0)
        self.progressBar_dim_0.setObjectName("progressBar_dim_0")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(70, 700, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(70, 740, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.progressBar_dim_1 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_dim_1.setGeometry(QtCore.QRect(230, 750, 311, 23))
        self.progressBar_dim_1.setProperty("value", 0)
        self.progressBar_dim_1.setObjectName("progressBar_dim_1")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(70, 780, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.progressBar_dim_2 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_dim_2.setGeometry(QtCore.QRect(230, 790, 311, 23))
        self.progressBar_dim_2.setProperty("value", 0)
        self.progressBar_dim_2.setObjectName("progressBar_dim_2")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(70, 820, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.progressBar_dim_3 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_dim_3.setGeometry(QtCore.QRect(230, 830, 311, 23))
        self.progressBar_dim_3.setProperty("value", 0)
        self.progressBar_dim_3.setObjectName("progressBar_dim_3")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(670, 700, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.progressBar_dim_8 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_dim_8.setGeometry(QtCore.QRect(830, 830, 311, 23))
        self.progressBar_dim_8.setProperty("value", 0)
        self.progressBar_dim_8.setObjectName("progressBar_dim_8")
        self.progressBar_dim_5 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_dim_5.setGeometry(QtCore.QRect(830, 710, 311, 23))
        self.progressBar_dim_5.setProperty("value", 0)
        self.progressBar_dim_5.setObjectName("progressBar_dim_5")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(670, 820, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(670, 780, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.progressBar_dim_6 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_dim_6.setGeometry(QtCore.QRect(830, 750, 311, 23))
        self.progressBar_dim_6.setProperty("value", 0)
        self.progressBar_dim_6.setObjectName("progressBar_dim_6")
        self.progressBar_dim_7 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_dim_7.setGeometry(QtCore.QRect(830, 790, 311, 23))
        self.progressBar_dim_7.setProperty("value", 0)
        self.progressBar_dim_7.setObjectName("progressBar_dim_7")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(670, 740, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        self.label_19.setGeometry(QtCore.QRect(70, 860, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.progressBar_dim_4 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_dim_4.setGeometry(QtCore.QRect(230, 870, 311, 23))
        self.progressBar_dim_4.setProperty("value", 0)
        self.progressBar_dim_4.setObjectName("progressBar_dim_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.lineEdit, self.pushButton)
        MainWindow.setTabOrder(self.pushButton, self.pushButton_2)
        MainWindow.setTabOrder(self.pushButton_2, self.lineEdit_2)
        MainWindow.setTabOrder(self.lineEdit_2, self.horizontalSlider)
        MainWindow.setTabOrder(self.horizontalSlider, self.horizontalSlider_2)
        MainWindow.setTabOrder(self.horizontalSlider_2, self.horizontalSlider_3)
        MainWindow.setTabOrder(self.horizontalSlider_3, self.horizontalSlider_4)
        MainWindow.setTabOrder(self.horizontalSlider_4, self.horizontalSlider_5)
        MainWindow.setTabOrder(self.horizontalSlider_5, self.pushButton_4)
        MainWindow.setTabOrder(self.pushButton_4, self.pushButton_3)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Mind Locker"))
        self.pushButton.setText(_translate("MainWindow", "开始录制脑纹"))
        self.pushButton_2.setText(_translate("MainWindow", "重新开始"))
        self.label.setText(_translate("MainWindow", "第一步： 请输入你的名字"))
        self.label_1.setText(_translate("MainWindow", "Mind Locker- 脑纹识别与加密技术"))
        self.pushButton_4.setText(_translate("MainWindow", "复位"))
        self.label_3.setText(_translate("MainWindow", "服务器地址"))
        self.label_4.setText(_translate("MainWindow", "EEG总采样帧数"))
        self.label_5.setText(_translate("MainWindow", "每批训练的帧数"))
        self.label_6.setText(_translate("MainWindow", "训练的总次数"))
        self.label_7.setText(_translate("MainWindow", "用于识别的帧组数"))
        self.label_8.setText(_translate("MainWindow", "每帧组的帧数"))
        self.lineEdit_2.setInputMask(_translate("MainWindow", "000.000.000.000"))
        self.lineEdit_2.setText(_translate("MainWindow", "127.0.0.1"))
        self.lineEdit.setText(_translate("MainWindow", "请输入你的名字"))
        self.pushButton_3.setText(_translate("MainWindow", "载入训练完成的脑纹"))
        self.pushButton_5.setText(_translate("MainWindow", "载入未训练的脑纹"))
        self.label_10.setText(_translate("MainWindow", "特征1的匹配度"))
        self.label_11.setText(_translate("MainWindow", "特征2的匹配度"))
        self.label_12.setText(_translate("MainWindow", "特征3的匹配度"))
        self.label_13.setText(_translate("MainWindow", "特征4的匹配度"))
        self.label_15.setText(_translate("MainWindow", "特征6的匹配度"))
        self.label_16.setText(_translate("MainWindow", "特征9的匹配度"))
        self.label_17.setText(_translate("MainWindow", "特征8的匹配度"))
        self.label_18.setText(_translate("MainWindow", "特征7的匹配度"))
        self.label_19.setText(_translate("MainWindow", "特征5的匹配度"))

