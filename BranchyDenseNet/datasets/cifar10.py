"""
load cifar10, return training set and testing set
X(,,,,) Y(,)
"""
from chainer.datasets import cifar
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    (train,test)=cifar.get_cifar10()
    x_train, y_train = process_data(train)
    x_test, y_test = process_data(test)
    # img=np.reshape(x_train[123],(3,32,32))
    # img=img.transpose(1,2,0)
    # plt.imshow(img)
    # plt.show()
    # print(y_train[123])

    return x_train,y_train,x_test,y_test

def process_data(data):
    x = []
    y = []
    for d in data:
        x.append(d[0])
        y.append(d[1])

    x = np.array(x)
    # x = x.reshape([-1, 1, 32, 32])
    y = np.array(y)

    return x, y
