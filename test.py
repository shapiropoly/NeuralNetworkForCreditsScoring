import numpy as np
from parse_data import processing_data
from constants import *

w1 = np.random.randn(INPUT_NEURONS, H_NEURONS)






(x_train, y_train), (x_test, y_test) = processing_data()


for x, y in zip(x_train, y_train):

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    print(np.dot(x, w1))

    # print(x_train[row])
    # print("row", row)
    # x = np.array(row)
    #
    # print("x", x)
    # y = y_train
    #
    #
    # t1 = np.dot(x, w1)
    #
    # print(t1)
    #

