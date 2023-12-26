"""
This example demonstrates how to use the active learning interface with Pytorch.
The example uses Skorch, a scikit learn wrapper of Pytorch.
For more info, see https://skorch.readthedocs.io/en/stable/
"""
import math
import torch.nn.functional as F

import modAL
from modAL.uncertainty import entropy_sampling
import numpy as np

import pandas as pd

import copy

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn import metrics

from scipy.spatial.distance import pdist, squareform

from collections import OrderedDict

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

import scipy.io as scio
DATA1 = 'data.npy'
TARGET1 = 'target.npy'
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from skorch import NeuralNetClassifier
import matplotlib.pyplot as plt

from modAL.models import ActiveLearner


# # build class for the skorch API
# class Torch_Model(nn.Module):
#     def __init__(self,):
#         super(Torch_Model, self).__init__()
#         self.convs = nn.Sequential(
#                                 nn.Conv2d(1,32,3),
#                                 nn.ReLU(),
#                                 nn.Conv2d(32,64,3),
#                                 nn.ReLU(),
#                                 nn.MaxPool2d(2),
#                                 nn.Dropout(0.25)
#         )
#         self.fcs = nn.Sequential(
#                                 nn.Linear(12*12*64,128),
#                                 nn.ReLU(),
#                                 nn.Dropout(0.5),
#                                 nn.Linear(128,10),
#         )
#
#     def forward(self, x):
#         out = x
#         out = self.convs(out)
#         out = out.view(-1,12*12*64)
#         out = self.fcs(out)
#         return out


class Torch_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(1200,128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Linear(64,5),
            nn.Tanh()
        )
    def forward(self, xb):
        return self.network(xb)


# create the classifier
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = NeuralNetClassifier(Torch_Model,
                                 # max_epochs=100,
                                 criterion=nn.CrossEntropyLoss,
                                 optimizer=torch.optim.Adam,
                                 train_split=None,
                                 verbose=1,
                                 device="cpu",lr=0.0001)

"""
Data wrangling
1. Reading data from torchvision
2. Assembling initial training data for ActiveLearner
3. Generating the pool
"""

def get_train_test(split_ratio=.6, random_state=42):
    X = np.load(DATA1)
    y = np.load(TARGET1)
    assert X.shape[0] == y.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state,
                                                        shuffle=True)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test=get_train_test()
X_train = X_train.reshape(-1,1200)
X_test = X_test.reshape(-1,1200)

# mnist_data = MNIST('.', download=True, transform=ToTensor())
# dataloader = DataLoader(mnist_data, shuffle=True, batch_size=60000)
# X, y = next(iter(dataloader))

# read training data
# X_train, X_test, y_train, y_test = X[:50000], X[50000:], y[:50000], y[50000:]
# X_train = X_train.reshape(50000, 1, 28, 28)
# X_test = X_test.reshape(10000, 1, 28, 28)

# assemble initial data
biaozhuset=[]
n_initial = 1000
initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
biaozhuset.extend(initial_idx)
X_initial=X_train[initial_idx]
y_initial = y_train[initial_idx]
# initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
# X_initial = X_train[initial_idx]
# y_initial = y_train[initial_idx]
# X_initial=X_initial.numpy()
# y_initial=y_initial.numpy()
# generate the pool
# remove the initial data from the training dataset
X_pool = np.delete(X_train, initial_idx, axis=0)
y_pool = np.delete(y_train, initial_idx, axis=0)
# X_pool = np.delete(X_train, initial_idx, axis=0)
# y_pool = np.delete(y_train, initial_idx, axis=0)
# X_pool=X_pool.numpy()
# y_pool=y_pool.numpy()

# """
# Training the ActiveLearner
# """
#
# # # initialize ActiveLearner
# learner = ActiveLearner(
#     estimator=classifier,
#     X_training=X_initial, y_training=y_initial
# )
#
# listac=[]
# listactest=[]
# # the active learning loop
# n_queries = 5
# for idx in range(n_queries):
#     query_idx, query_instance = learner.query(X_pool, n_instances=len(X_pool))
#
#     learner.teach(X_pool[query_idx], y_pool[query_idx])#,only_new=True
#     # # remove queried instance from pool
#     # X_pool = np.delete(X_pool, query_idx, axis=0)
#     # y_pool = np.delete(y_pool, query_idx, axis=0)
#     listac.append(learner.score(X_train, y_train))
#     listactest.append(learner.score(X_test, y_test))
#
# # the final accuracy score
# print(learner.score(X_test, y_test))
# plt.plot([i for i in range(len(listac))],listac,c='blue',label = 'AL_ALL_train_ac1')
# plt.plot([i for i in range(len(listactest))],listactest,c='blue',label = 'AL_ALL_test_ac2')
# plt.plot([i for i in range(len(listac))],listac,'.',c='blue')
# plt.plot([i for i in range(len(listactest))],listactest,'.',c='blue')
#
#
#
#
#
def random_sampling( X_pool,n_instances):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples),size=n_instances,replace=True)
    return query_idx, X_pool[query_idx]
#
# # initialize ActiveLearner
# learner = ActiveLearner(
#     estimator=classifier,
#     query_strategy=random_sampling,
#     X_training=X_initial, y_training=(y_initial)
# )
#
#
# listac=[]
# listactest=[]
# X_pool_random_sampling=X_pool
# y_pool_random_sampling=y_pool
# # the active learning loop
# n_queries = 5
# for idx in range(n_queries):
#     query_idx, query_instance = learner.query(X_pool_random_sampling, n_instances=1000)
#     learner.teach(X_pool_random_sampling[query_idx], y_pool_random_sampling[query_idx])#,only_new=True,only_new=True
#     # remove queried instance from pool
#     # X_pool_entropy_sampling = np.delete(X_pool_random_sampling, query_idx, axis=0)
#     # y_pool_entropy_sampling = np.delete(X_pool_random_sampling, query_idx, axis=0)
#     listac.append(learner.score(X_train, y_train))
#     listactest.append(learner.score(X_test, y_test))
#
#
# # the final accuracy score
# print(learner.score(X_test, y_test))
# plt.plot([i for i in range(len(listac))],listac, c='red',label = 'AL_random_train_ac1')
# plt.plot([i for i in range(len(listactest))],listactest, c='red',label = 'AL_random_test_ac2')
#
#
#
#
#
#
class DPC(object):

    def __init__(self, X, clusterNum, distPercent):

        self.X = X

        self.N = X.shape[0]

        self.clusterNum = clusterNum

        self.distPercent = distPercent

        self.distCut = 0

        self.rho = np.zeros(self.N, dtype=float)

        self.delta = np.zeros(self.N, dtype=float)

        self.gamma = np.zeros(self.N, dtype=float)

        self.leader = np.ones(self.N, dtype=int) * int(-1)

        self.distList = pdist(self.X, metric='euclidean')

        self.distMatrix = squareform(self.distList)

        self.clusterIdx = np.ones(self.N, dtype=int) * (-1)

    def getDistCut(self):

        maxDist = max(self.distList)

        distCut = maxDist * self.distPercent / 100

        return distCut

    def getRho(self):

        self.distCut = self.getDistCut()

        rho = np.zeros(self.N, dtype=float)

        for i in range(self.N - 1):

            for j in range(i + 1, self.N):

                if self.distMatrix[i, j] < self.distCut:
                    rho[i] += 1

                    rho[j] += 1

        return rho

    def getGammaOrderIndex(self):

        self.rho = self.getRho()

        rhoOrdIndex = np.flipud(np.argsort(self.rho))

        # -----------获取块密度最大点的Delta----------------#

        maxdist = 0

        for i in range(self.N):

            if self.distMatrix[rhoOrdIndex[0], i] > maxdist:
                maxdist = self.distMatrix[rhoOrdIndex[0], i]

        self.delta[rhoOrdIndex[0]] = maxdist

        self.leader[rhoOrdIndex[0]] = -1

        # -----------获取非密度最大点的Delta----------------#

        for i in range(1, self.N):

            mindist = np.inf

            minindex = -1

            for j in range(i):

                if self.distMatrix[rhoOrdIndex[i], rhoOrdIndex[j]] < mindist:
                    mindist = self.distMatrix[rhoOrdIndex[i], rhoOrdIndex[j]]

                    minindex = rhoOrdIndex[j]

            self.delta[rhoOrdIndex[i]] = mindist

            self.leader[rhoOrdIndex[i]] = minindex

        self.gamma = self.delta * self.rho

        gammaOrderIndex = np.flipud(np.argsort(self.gamma))

        return gammaOrderIndex, rhoOrdIndex

    def getDPC(self):

        gammaOrderIndex, rhoOrdIndex = self.getGammaOrderIndex()
        # print(rhoOrdIndex)

        # -----------给聚类中心分配簇标签------------------#

        for i in range(self.clusterNum):
            self.clusterIdx[gammaOrderIndex[i]] = i

        # # --------开始聚类-----------------------#
        #
        # for i in range(self.N):
        #
        #     if self.clusterIdx[rhoOrdIndex[i]] == -1:
        #         self.clusterIdx[rhoOrdIndex[i]] = self.clusterIdx[self.leader[rhoOrdIndex[i]]]
        #
        # ##-------------初始化一个空字典，用于存储类簇---------------##
        #
        # clusterSet = OrderedDict()
        #
        # # --------字典初始化，使用列表存储类簇-----------#
        #
        # for i in range(self.clusterNum):
        #     clusterSet[i] = []
        #
        # # ---将每个样本根据类簇标号分配到字典当中---#
        #
        # for i in range(self.N):
        #     clusterSet[self.clusterIdx[i]].append(i)

        return gammaOrderIndex[:self.clusterNum]
#initialize ActiveLearner

'''计算已知数据的熵'''


###########################################

def DataEntropy(data):

    entropyVal = 0;
    for i in range((len(data))):
        proptyVal = data[i]

        entropyVal = entropyVal - proptyVal * math.log2(proptyVal)

    return entropyVal


def maxshang(X_pool,X_pool_pppp,n_instances):
    tmplsit = []
    for i in range(len(X_pool_pppp)):
        entropyVal = DataEntropy(X_pool_pppp[i])
        tmp = {"index": i, "V": entropyVal}
        tmplsit.append(tmp)
    tmplsit.sort(key=lambda stu: stu["V"])
    query_idx = []
    for i in range(n_instances):
        query_idx.append(tmplsit[i]['index'])
    return query_idx, X_pool[query_idx]


def dalget(X_pool,falg="ENT"):
    c=500
    k = 2 * c
    if falg=="ENT":

        output = model(torch.tensor(X_pool))
        aaa = F.softmax(output).detach().numpy()
        # aaa = learner.predict_proba(X_pool)
        query_idx, query_instance = maxshang(X_pool_pppp=aaa, X_pool=X_pool, n_instances=k)
    else:
        pass
    dpc = DPC(X_pool[query_idx], clusterNum=c, distPercent=7)
    clusterId_C = dpc.getDPC()
    query_idx=clusterId_C
    dhal=query_idx
    return dhal


def dhget(X_pool_zs,biaojie):
    rh1=0.8
    rh2=0.6
    yuzhi=0.4
    if len(biaojie)/len(X_train)>yuzhi:
        kj=int(rh1*len(biaojie))
        dh1=[]
        dpc = DPC(X_pool_zs[biaojie], clusterNum=kj, distPercent=7)
        clusterId_C = dpc.getDPC()
        dh1.extend(clusterId_C)

        output = model(torch.tensor(X_pool_zs[dh1]))
        aaa = F.softmax(output).detach().numpy()
        #!!!!aaa = learner.predict_proba(X_pool)
        query_idx, query_instance = maxshang(X_pool_pppp=aaa, X_pool=X_pool_zs[biaojie], n_instances=int(rh2*len(dh1)))
        dh=query_idx
    else:
        dh=biaojie
    return dh

#
#
#
#
# initialize ActiveLearner
# learner = ActiveLearner(
#     estimator=classifier,
#     X_training=X_initial, y_training=y_initial
# )
#
#
# listac=[]
# listactest=[]
# X_pool_zs=X_train
# y_pool_zs=y_train
# # # the active learning loop
# # n_queries = 5
# #
# # for idx in range(n_queries):
# #     # s=1000
# #     indexset_zs=[]
# #     dhal=dalget(X_pool_zs)
# #
# #     biaozhuset.extend(dhal)
# #     indexset_zs.extend(dhal)
# #     # print(len(indexset_zs))
# #
# #     dh=dhget(X_pool_zs,biaozhuset)
# #     indexset_zs.extend(dh)
# #
# #     biaozhuset.extend(indexset_zs)
# #     biaozhuset=set(biaozhuset)
# #     biaozhuset=list(biaozhuset)
# #     print(len(indexset_zs))
# #     # print(len(indexset_zs))
# #     # query_idx, query_instance = learner.query(X_pool_random_sampling, n_instances=10)
# #     learner.teach(X_train[indexset_zs], y_train[indexset_zs])#,only_new=True
# #     # remove queried instance from pool
# #     # X_pool_entropy_sampling = np.delete(X_pool_zs, indexset_zs, axis=0)
# #     # y_pool_entropy_sampling = np.delete(y_pool_zs, indexset_zs, axis=0)
# #     listac.append(learner.score(X_train, y_train))
# #     listactest.append(learner.score(X_test, y_test))
# #
# #
# # # the final accuracy score
# # print(learner.score(X_test, y_test))
# # plt.plot([i for i in range(len(listac))],listac,c='green',label = 'AL_ENT_DCP_ac')
# # plt.plot([i for i in range(len(listactest))],listactest,c='green',label = 'AL_ENT_DCP_test_ac')
# # plt.plot([i for i in range(len(listac))],listac,'*',c='green')
# # plt.plot([i for i in range(len(listactest))],listactest,'*',c='green')
# #
# # plt.xlabel('times')
# # plt.ylabel('ac')
# # plt.title('ac-learning')
# # plt.show()
#
#
#
#
#
#
model=Torch_Model()

#
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_func = torch.nn.CrossEntropyLoss()
#
#

listac=[]
listactest=[]
X_pool_zs=X_train
y_pool_zs=y_train
for t in range(200):
    # print(X_train[t].reshape(-1, ).shape)
    indexset_zs = []
    dhal = dalget(X_pool_zs)

    biaozhuset.extend(dhal)
    indexset_zs.extend(dhal)
    # print(len(indexset_zs))

    dh = dhget(X_pool_zs, biaozhuset)
    indexset_zs.extend(dh)

    biaozhuset.extend(indexset_zs)
    biaozhuset = set(biaozhuset)
    biaozhuset = list(biaozhuset)
    print(len(indexset_zs))
    out = model(torch.tensor(X_train[indexset_zs]))  # input x and predict based on x  # 喂给 net 训练数据 x, 输出预测值
    loss = F.cross_entropy(out, torch.tensor(y_train[indexset_zs]).long())  # must be (1. nn output, 2. target), the target label is NOT one-hotted  # 计算两者的误差

    optimizer.zero_grad()  # clear gradients for next train      # 清空上一步的残余更新参数值
    loss.backward()  # backpropagation, compute gradients  # 误差反向传播, 计算参数更新值
    optimizer.step()  # apply gradients                     # 将参数更新值施加到 net 的 parameters 上

    if t %  10  == 0:
        # plot and show learning process
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        # prediction = torch.max(out, 1)[1]
        # pred_y = torch.tensor(prediction).data
        output = model(torch.tensor(X_train))
        pred=[]
        probs = F.softmax(output).detach().numpy()
        for i in range(len(probs)):
            tm1 = np.argmax(probs[i])
            pred.append(tm1)
        target_y = torch.tensor(y_train).data
        tmpac=int(((torch.tensor(pred) == target_y).sum()).numpy().tolist())
        accuracy = float(tmpac) / float(len(target_y))  # 预测中有多少和真实值一样
        listac.append(accuracy)
        print('Accuracy=%.2f' % accuracy)

        output = model(torch.tensor(X_test))
        pred=[]
        probs = F.softmax(output).detach().numpy()
        for i in range(len(probs)):
            tm1 = np.argmax(probs[i])
            pred.append(tm1)
        target_y = torch.tensor(y_test).data
        tmpac=int(((torch.tensor(pred) == target_y).sum()).numpy().tolist())
        accuracy = float(tmpac) / float(len(target_y))  # 预测中有多少和真实值一样
        listactest.append(accuracy)
        print('Accuracy=%.2f' % accuracy)

x=[i for i in range(len(listac))]
plt.plot(x,listac)

x=[i for i in range(len(listactest))]
plt.plot(x,listactest)
# plt.show()





model=Torch_Model()

#
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_func = torch.nn.CrossEntropyLoss()
#
#

listac=[]
listactest=[]
X_pool_zs=X_train
y_pool_zs=y_train

for t in range(200):
    # print(X_train[t].reshape(-1, ).shape)
    query_idx, query_instance = random_sampling(X_pool_zs, n_instances=2000)

    out = model(torch.tensor(X_train[query_idx]))  # input x and predict based on x  # 喂给 net 训练数据 x, 输出预测值
    loss = F.cross_entropy(out, torch.tensor(y_train[query_idx]).long())  # must be (1. nn output, 2. target), the target label is NOT one-hotted  # 计算两者的误差

    optimizer.zero_grad()  # clear gradients for next train      # 清空上一步的残余更新参数值
    loss.backward()  # backpropagation, compute gradients  # 误差反向传播, 计算参数更新值
    optimizer.step()  # apply gradients                     # 将参数更新值施加到 net 的 parameters 上

    if t %10 == 0:
        # plot and show learning process
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        # prediction = torch.max(out, 1)[1]
        # pred_y = torch.tensor(prediction).data
        output = model(torch.tensor(X_train))
        pred=[]
        probs = F.softmax(output).detach().numpy()
        for i in range(len(probs)):
            tm1 = np.argmax(probs[i])
            pred.append(tm1)
        target_y = torch.tensor(y_train).data
        tmpac=int(((torch.tensor(pred) == target_y).sum()).numpy().tolist())
        accuracy = float(tmpac) / float(len(target_y))  # 预测中有多少和真实值一样
        listac.append(accuracy)
        print('Accuracy=%.2f' % accuracy)

        output = model(torch.tensor(X_test))
        pred=[]
        probs = F.softmax(output).detach().numpy()
        for i in range(len(probs)):
            tm1 = np.argmax(probs[i])
            pred.append(tm1)
        target_y = torch.tensor(y_test).data
        tmpac=int(((torch.tensor(pred) == target_y).sum()).numpy().tolist())
        accuracy = float(tmpac) / float(len(target_y))  # 预测中有多少和真实值一样
        listactest.append(accuracy)
        print('Accuracy=%.2f' % accuracy)

x=[i for i in range(len(listac))]
plt.plot(x,listac)

x=[i for i in range(len(listactest))]
plt.plot(x,listactest)




model = Torch_Model()

#
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_func = torch.nn.CrossEntropyLoss()
#
#

listac = []
listactest = []
X_pool_zs = X_train
y_pool_zs = y_train

for t in range(20):
    # print(X_train[t].reshape(-1, ).shape)

    out = model(torch.tensor(X_train))  # input x and predict based on x  # 喂给 net 训练数据 x, 输出预测值
    loss = F.cross_entropy(out, torch.tensor(
        y_train).long())  # must be (1. nn output, 2. target), the target label is NOT one-hotted  # 计算两者的误差

    optimizer.zero_grad()  # clear gradients for next train      # 清空上一步的残余更新参数值
    loss.backward()  # backpropagation, compute gradients  # 误差反向传播, 计算参数更新值
    optimizer.step()  # apply gradients                     # 将参数更新值施加到 net 的 parameters 上

    if t % 10 == 0:
        # plot and show learning process
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        # prediction = torch.max(out, 1)[1]
        # pred_y = torch.tensor(prediction).data
        output = model(torch.tensor(X_train))
        pred = []
        probs = F.softmax(output).detach().numpy()
        for i in range(len(probs)):
            tm1 = np.argmax(probs[i])
            pred.append(tm1)
        target_y = torch.tensor(y_train).data
        tmpac = int(((torch.tensor(pred) == target_y).sum()).numpy().tolist())
        accuracy = float(tmpac) / float(len(target_y))  # 预测中有多少和真实值一样
        listac.append(accuracy)
        print('Accuracy=%.2f' % accuracy)

        output = model(torch.tensor(X_test))
        pred = []
        probs = F.softmax(output).detach().numpy()
        for i in range(len(probs)):
            tm1 = np.argmax(probs[i])
            pred.append(tm1)
        target_y = torch.tensor(y_test).data
        tmpac = int(((torch.tensor(pred) == target_y).sum()).numpy().tolist())
        accuracy = float(tmpac) / float(len(target_y))  # 预测中有多少和真实值一样
        listactest.append(accuracy)
        print('Accuracy=%.2f' % accuracy)

x = [i for i in range(len(listac))]
plt.plot(x, listac)

x = [i for i in range(len(listactest))]
plt.plot(x, listactest)

plt.show()
