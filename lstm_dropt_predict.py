# -*- coding: utf-8 -*-

import numpy as np   #用于处理数组和矩阵。
import pandas as pd   # 用于数据处理和分析，这里主要用于将 NumPy 数组转换为 DataFrame。
import seaborn as sns  # 用于数据可视化，特别是热力图。
import matplotlib.pyplot as plt
from keras.layers import MaxPooling2D,Dropout,Flatten
import chinese_calendar
import warnings
warnings.filterwarnings('ignore')     #屏蔽掉Python中产生的所有警告信息，某些库或者函数可能会产生不影响程序执行的警告信息，屏蔽它们可以让输出更加清晰。
from sklearn.preprocessing import MinMaxScaler    #从sklearn.preprocessing中导入，用于数据的归一化处理，将数据缩放到[0, 1]的范围内。
from keras.models import Sequential
from keras.layers import Dense, LSTM    #用于构建和定义神经网络模型
import heartpy as hp   #用于处理和分析心电图（ECG）数据的库
import matplotlib.pyplot as plt    #用于数据可视化
import scipy.io as sio   #用于科学计算和信号处理
from scipy import signal as sig
import numpy as np    #用于高效的数值计算
from sklearn.metrics import mean_squared_error, r2_score
plt.rcParams['font.family']='SimHei' # 中文乱码
plt.rcParams['axes.unicode_minus']=False # 负号无法正常显示

"""
特征归一化函数，它采用输入数据 X，通常是一个二维的 NumPy 数组，其中每一行表示一个数据点，每一列表示一个特征，并对数据进行标准化处理。
"""
def feature_normalize(X):
    mean = np.mean(X, axis=0)      #计算每个特征（即每列）的平均值，axis=0 表示沿着行的方向（即对每个特征的所有数据点）计算平均值。
    std = np.std(X, axis=0)        #计算每个特征的标准差。标准差是衡量数据分布宽度的统计量，axis=0 同样表示沿着行的方向计算。
    return (X - mean) / std        #对数据进行标准化处理：(X - mean) / std，即 z-score 标准化或规范化。每个数据点减去其所在特征的平均值，然后除以其所在特征的标准差

if __name__ == "__main__":
    # 利用heartpy库对心电图或PPG信号进行加载、可视化、处理和特征提取.
    #hp.load_exampledata(0)和hp.load_exampledata(1)分别加载了两组不同的心电图或PPG示例数据。这些数据包含了一段时间内心脏的电活动或血流变化信息。
    #对于第一组数据，采样率被设定为固定的100.0Hz，因为该组数据的采样率已知且恒定。
    #对于第二组数据，采样率是通过hp.get_samplerate_mstimer(timer)从提供的时间戳信息中计算得出的，这反映了实际数据采集时的采样频率。

    data, timer = hp.load_exampledata(0)   #使用hp.load_exampledata(0)加载第一组示例数据
    #使用matplotlib库来绘制信号的波形图。这有助于直观地查看信号的原始形态，从而对其质量进行初步评估，并识别任何异常或噪声.
    plt.figure(figsize=(12, 4))   #对于每一组数据，使用matplotlib库,通过plt创建一个大小为12x4的图形窗口，并绘制出加载的信号数据。
    plt.plot(data)

    wd, m = hp.process(data, sample_rate=100.0)   #对于第一组数据,使用固定的采样率（100.0 Hz）对信号数据进行处理，hp.process函数会返回处理后的波形数据wd和一系列计算出的度量m。
    plt.figure(figsize=(12, 4))   #创建一个新的图形窗口，并使用hp.plotter函数将处理后的波形数据和检测到的峰值等信息绘制出来。
    hp.plotter(wd, m)           #绘制处理后的信号及其分析结果
    for measure in m.keys():
        print('%s: %f' % (measure, m[measure]))    #遍历度量字典m，并打印出每个度量的名称和值

    #类似地，加载第二个示例数据，并绘制原始信号。
    data, timer = hp.load_exampledata(1)    #加载第二组示例数据
    plt.figure(figsize=(12, 4))
    plt.plot(data)

    sample_rate = hp.get_samplerate_mstimer(timer)  #对于第二组数据，采样率不是固定的，而是通过hp.get_samplerate_mstimer(timer)根据时间戳计算得出
    wd, m = hp.process(data, sample_rate)
    # plot
    plt.figure(figsize=(12, 4))    #创建一个新的图形窗口，并使用hp.plotter函数将处理后的波形数据和检测到的峰值等信息绘制出来。
    hp.plotter(wd, m)    #绘制处理后的信号及其分析结果
    for measure in m.keys():
        print('%s: %f' % (measure, m[measure]))    #遍历度量字典m，并打印出每个度量的名称和值

   #加载第三个示例数据，并打印出第一个时间戳，以了解数据开始的时间。
    data, timer = hp.load_exampledata(2)
    print(timer[0])
    sample_rate = hp.get_samplerate_datetime(timer, timeformat='%Y-%m-%d %H:%M:%S.%f')
    print('sample rate is: %f Hz' % sample_rate)

    wd, m = hp.process(data, sample_rate, report_time=True)   #这次的时间戳是日期时间格式，所以使用hp.get_samplerate_datetime函数和指定的时间格式来计算采样率
    plt.figure(figsize=(12, 4))
    hp.plotter(wd, m)

    plt.figure(figsize=(12, 4))
    plt.xlim(20000, 30000)   #创建一个新的图形窗口，并设置x轴的范围为20,000到30,000，以便更详细地查看这个特定区间的波形
    hp.plotter(wd, m)
    for measure in m.keys():
        print('%s: %f' % (measure, m[measure]))
    plt.show()   #plt.show()函数被调用，以显示所有之前创建的图形窗口，确保所有的图形都被正确地渲染和显示。

    import tensorflow as tf

    matpath = './'   #设置 MATLAB 文件的存储路径为当前目录。
    filename_sp = 'systolicpres.mat'  #设置包含收缩压数据的文件名。
    filename_params = 'ppg_params.mat'  #设置包含 PPG（光电容积描记图）参数数据的文件名。

    systolic_pressure = sio.loadmat(matpath + filename_sp)  #加载收缩压数据
    ppg_params = sio.loadmat(matpath + filename_params)    # 加载 PPG 参数数据


    y_data = np.transpose(systolic_pressure.get('BP_l'))   #从 `systolic_pressure` 数据中提取 'BP_l' 键对应的数组，并对其进行转置。这通常是因为 MATLAB 是按列优先存储的，而 Python/NumPy 是按行优先的。
    # y_train = y_data[0:500, :]
    # y_test = y_data[500:, :]

    x_data = np.transpose(ppg_params.get('PeakSys_l'))   # 类似地，从 `ppg_params` 数据中提取 'PeakSys_l' 键对应的数组并进行转置。
    x_data = feature_normalize(x_data)   #对 `x_data` 进行特征标准化。
    # x_train = x_data[0:500, :]
    # x_test = x_data[500:, :]

    feature=pd.DataFrame(x_data)  #使用 pandas 的 `corr()` 方法来计算特征之间的相关系数。
    corr = feature.corr()   #计算 DataFrame `feature` 中各列之间的相关系数。

    # 特征矩阵热力图可视化
    plt.figure(figsize=(10,6))
    ax = sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns, linewidths=0.2, cmap="RdYlGn",annot=True)
    plt.title("变量间相关系数")     #热力图中的颜色深浅表示了特征之间相关性的强弱。

    x = np.array(x_data)
    y = np.array(y_data)


    #划分训练集和测试集
    #需要注意的是，这里的模型是一个时间序列问题，
    #需要用前面的时间数据来预测后面的时间序列问题，
    #故在此用前90%作为训练集，后10%作为测试集。
    #而不能用train_test_split方法，对训练集和测试集进行随机划分。

    #数据归一化
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x = scaler_x.fit_transform(x)    # 使用MinMaxScaler对自变量x进行归一化
    y = scaler_y.fit_transform(np.reshape(y,(len(y),1)))   # 对因变量y进行归一化，并将其形状调整为二维列向量，数据缩放到【0，1】范围内


    #数据分割
    x_length = x.shape[0]
    split = int(x_length*0.8)   # # 划分训练集和测试集，比例为80%和20%
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    #数据形状调整
    # 由于LSTM的输入通常需要三维的形状（样本数, 时间步长, 特征数），这里将x_train和x_test的形状调整为(样本数, 特征数, 1)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

    #构建LSTM模型
    #使用Keras构建了一个简单的LSTM模型。模型包含一个LSTM层（32个神经元），一个Dropout层（防止过拟合），和一个全连接层（输出一个神经元）。模型使用均方误差作为损失函数，并使用Adam优化器进行优化。
    model = Sequential()
    model.add(LSTM(32, input_dim=1))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    #模型训练
    #使用训练数据对模型进行训练，总共训练1000个周期，每个批次包含100个样本。
    history =model.fit(x_train, y_train, epochs=1000, batch_size=100, verbose=2)

    #绘制损失曲线
    loss = history.history['loss']
    plt.figure()
    plt.plot(loss)
    plt.legend(['loss'])
    plt.xlabel('epoach')

    # output_train是模型在训练集上的预测值，y_train是训练集上的真实值。
    output_train = model.predict(x_train)
    output_test = model.predict(x_test)

    # 训练集性能评估和可视化
    # 首先，使用matplotlib的plot函数绘制了这两个序列，以便进行可视化比较。
    # 然后，计算了训练集上的RMSE和R²，并打印了这些值。
    plt.figure()   # 创建一个新的图形窗口
    plt.plot(range(len(output_train)), output_train, color='b', label='y_trainpre')   #绘制模型在训练集上的预测值（蓝色）
    plt.plot(range(len(output_train)),y_train, color='r', label='y_true')    # 绘制训练集上的真实值（红色）
    plt.legend()   # 显示图例
    train_rmse = np.sqrt(mean_squared_error(output_train,y_train))   # 计算训练集上的均方根误差（RMSE）
    train_r2 = r2_score(output_train, y_train)   # 计算训练集上的决定系数（R²）
    print('train_rmse = ' + str(round(train_rmse, 5)))   # 打印训练集上的RMSE，保留5位小数
    print('r2 = ', str(train_r2))  # 打印训练集上的R²

    # 测试集性能评估和可视化
    # 与训练集的处理类似，这里绘制了模型在测试集上的预测值和真实值,并计算了测试集上的RMSE和R²。
    # 这些指标提供了模型在未见过的数据上的泛化能力的信息。
    plt.figure()    # 创建一个新的图形窗口（注意：这里创建了一个新的图形窗口，而不是在之前的窗口上继续绘制）
    plt.plot(range(len(output_test)), output_test, color='b', label='y_testpre')
    plt.plot(range(len(output_test)),y_test, color='r', label='y_true')
    plt.legend()
    test_rmse = np.sqrt(mean_squared_error(output_test, y_test))
    test_r2 = r2_score(output_test,y_test)
    print('test_rmse = ' + str(round(test_rmse, 5)))
    print('r2 = ', str(test_r2))

    plt.show()   # 显示所有创建的图形窗口
    # results = np.hstack((y_test, output_test))
    # results = pd.DataFrame(results)
    # results.to_excel('SAE.xlsx')
