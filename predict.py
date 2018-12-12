# -*- coding: utf-8 -*-
# 評価
#

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from simple_net import SimpleNet
from util_dt import *
import time
import pickle

#
if __name__ == '__main__':
    # 学習データ
    global_start_time = time.time()
    #
    rdDim = pd.read_csv("sensors.csv", names=('id', 'temp', 'time') )
    fDim = rdDim["temp"]
    y_train = np.array(fDim, dtype = np.float32).reshape(len(fDim),1)
    x_train = conv_obj_dtArr(rdDim["time"] )
    #add N day
    x_test_pred = add_date_arr(rdDim["time"], 24 * 1 )
    n_train = int(len(x_train) * 0.1 )
    x_test = x_train[ n_train : ]
    y_test = y_train[ n_train : ]
    N= len(x_train)
    N_test  =len(x_test )
    num_max_y =100
    y_train =y_train / num_max_y
    y_test  =y_test / num_max_y
    print(x_train.shape, y_train.shape )
    print(x_test.shape  , y_test.shape )
    # load
    network = SimpleNet(input_size=1 , hidden_size=10, output_size=1 )
    network.load_params("params.pkl" )
    #print( network.params["W1"] )
    #pred
    train_acc = network.accuracy(x_train, y_train)
    test_acc  = network.accuracy(x_test, y_test)
    #
    print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc)   )
    #
    x_test_dt= conv_num_date(x_test_pred )
    x_train_dt= conv_num_date(x_train )
    #print(x_test_dt.shape )
    y_val = network.predict(x_test_pred )
    y_train = y_train * num_max_y
    y_val   = y_val * num_max_y    
    print ('time : ', time.time() - global_start_time)
    #print(y_val[:10] )
    #print(x_test_dt[:10] )
    #quit()
    #plt
    plt.plot(x_train_dt, y_train, label = "temp")
    plt.plot(x_test_dt , y_val , label = "predict")
    plt.legend()
    plt.grid(True)
    plt.title("IoT data")
    plt.xlabel("x_test")
    plt.ylabel("temperature")
    plt.show()
