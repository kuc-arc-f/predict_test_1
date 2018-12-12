# -*- coding: utf-8 -*-
# train/学習処理。結果ファイル保存。
# TwoLayerNet を参考に、３層ネットワーク利用
#  学習　>パラメータ保存

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from simple_net import SimpleNet
from util_dt import *
import time

#
if __name__ == '__main__':
    # 学習データ
    rdDim = pd.read_csv("sensors.csv", names=('id', 'temp', 'time') )
#    sort_arr  = rdDim.sort_values(by='time')
#    print(sort_arr[: 10])
#    quit()
    fDim = rdDim["temp"]
    #print(fDim[:10] )
    #quit()
    y_train = np.array(fDim, dtype = np.float32).reshape(len(fDim),1)
    #print(y_train[:20] )
    #quit()
    #print(fDim )
    #xDim =np.arange(len(fDim))
    #x_train =np.array(xDim, dtype = np.float32).reshape(len(xDim),1)
    x_train = conv_obj_dtArr(rdDim["time"] )
#    aa = add_date_arr(rdDim, 24 * 10 )
    #add N day
    x_test_pred = add_date_arr(rdDim["time"], 24 * 1 )
#    quit()
    #print(x_train[: 10] )
    #quit()
    # test-data
    #n_train = int(len(x_train) * 0.9)
    n_train = int(len(x_train) * 0.1 )
    x_test = x_train[ n_train : ]
    y_test = y_train[ n_train : ]
#    x_test_pred =get_pred_dat(x_test, 30 )

    #print( x_test[(len(x_test ) - 10) :] )
    #quit()
    N= len(x_train)
    N_test  =len(x_test )
    num_max_y =100
    y_train =y_train / num_max_y
    y_test  =y_test / num_max_y
    print(x_train.shape, y_train.shape )
    print(x_test.shape  , y_test.shape )
    #print(x_train[:10] )
    #print(x_test[:10] )
    #quit()
    #
    network = SimpleNet(input_size=1 , hidden_size=10, output_size=1 )
    iters_num = 3000  # 繰り返しの回数を適宜設定する    
    train_size = x_train.shape[0]
    print( train_size )
    #
    global_start_time = time.time()

#    batch_size = 100
    batch_size = 32
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

#    iter_per_epoch = max(train_size / batch_size, 1)
    iter_per_epoch =200
    #print(iter_per_epoch)
    #quit()

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = y_train[batch_mask]
        
        # 勾配の計算
        grad = network.gradient(x_batch, t_batch)
        
        # パラメータの更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, y_train)
            test_acc  = network.accuracy(x_test, y_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("i=" +str(i) + ", train acc, test acc | " + str(train_acc) + ", " + str(test_acc) + " , loss=" +str(loss) )
            print ('time : ', time.time() - global_start_time)
            #print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
    #pred
    train_acc = network.accuracy(x_train, y_train)
    test_acc  = network.accuracy(x_test, y_test)
    #
    print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc) + " , loss=" +str(loss) )
    print ('time : ', time.time() - global_start_time)
    #
    # パラメータの保存
    network.save_params("params.pkl")
    print("Saved Network Parameters!")
