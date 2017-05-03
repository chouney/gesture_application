# -*- coding:utf-8 -*-
import numpy as np
import os
from hmmlearn import hmm
import json
from sklearn import preprocessing
import warnings
import time
from itertools import izip
import zqx_utils
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
warnings.filterwarnings("ignore")

np.random.seed(42)


def store(data):
    with open('feature', 'w') as json_file:
        json_file.write(json.dumps(data))


def load(filepack, prefilename="feature"):
    datas = []
    lens = []
    files = os.listdir(filepack)
    for name in files:
        if name.startswith(prefilename):
            sample = 0
            with open(filepack + "/" + name) as json_file:
                data = json.load(json_file)
                for row in data:
                    datas.append(row)
                    sample += 1
            lens.append(sample)
    return preprocessing.normalize(datas), lens

# 获得神经网络的输入结点的集合，如果手势被拒识模型识别，则插入-1
def hmm_score_get_test_nn(gesture, feature="feature"):
    test_isTurn, lenT = load(gesture + "/isTurn", feature)
    test_normal, lenN = load(gesture + "/normal", feature)
    test_radis, lenR = load(gesture + "/radis", feature)
    test_velocity, lenV = load(gesture + "/velocity", feature)
    out = []
    isTurn = []
    normal = []
    radis = []
    velocity = []
    trainIndex = 0
    trainLen = lenT[trainIndex]
    index = 0
    # 获取神经网路的训练集
    for t, n, r, v in izip(test_isTurn, test_normal, test_radis, test_velocity):
        if index >= trainLen:
            index = 0
            trainIndex += 1
            if trainIndex < len(lenT):
                trainLen = lenT[trainIndex]
            # 计算该组训练集的score
            res = []
            Z1 = circle_isTurn_model.score(isTurn)
            Z2 = cross_isTurn_model.score(isTurn)
            Z3 = left_slide_isTurn_model.score(isTurn)
            Z4 = right_slide_isTurn_model.score(isTurn)
            Z5 = unrecogonized_isTurn_model.score(isTurn)
            print "isturn 模型打分",Z1,Z2,Z3,Z4,Z5
            res.append(Z1)
            res.append(Z2)
            res.append(Z3)
            res.append(Z4)
            Z1 = circle_normal_model.score(normal)
            Z2 = cross_normal_model.score(normal)
            Z3 = left_slide_normal_model.score(normal)
            Z4 = right_slide_normal_model.score(normal)
            Z5 = unrecogonized_normal_model.score(normal)
            print "normal 模型打分",Z1,Z2,Z3,Z4,Z5
            res.append(Z1)
            res.append(Z2)
            res.append(Z3)
            res.append(Z4)
            Z1 = circle_radius_model.score(radis)
            Z2 = cross_radius_model.score(radis)
            Z3 = left_slide_radius_model.score(radis)
            Z4 = right_slide_radius_model.score(radis)
            Z5 = unrecogonized_radius_model.score(radis)
            print "radius 模型打分",Z1,Z2,Z3,Z4,Z5
            res.append(Z1)
            res.append(Z2)
            res.append(Z3)
            res.append(Z4)
            Z1 = circle_velocity_model.score(velocity)
            Z2 = cross_velocity_model.score(velocity)
            Z3 = left_slide_velocity_model.score(velocity)
            Z4 = right_slide_velocity_model.score(velocity)
            Z5 = unrecogonized_velocity_model.score(velocity)
            print "velocity 模型打分",Z1,Z2,Z3,Z4,Z5
            res.append(Z1)
            res.append(Z2)
            res.append(Z3)
            res.append(Z4)
            normal_res = preprocessing.normalize(res)
            out.append(list(normal_res[0]))
            isTurn = []
            normal = []
            radis = []
            velocity = []
        # 拼接每组的训练集结果
        isTurn.append(t)
        normal.append(n)
        radis.append(r)
        velocity.append(v)
        index += 1
    res = []
    Z1 = circle_isTurn_model.score(isTurn)
    Z2 = cross_isTurn_model.score(isTurn)
    Z3 = left_slide_isTurn_model.score(isTurn)
    Z4 = right_slide_isTurn_model.score(isTurn)
    Z5 = unrecogonized_isTurn_model.score(isTurn)
    print "isturn 模型打分", Z1, Z2, Z3, Z4, Z5
    res.append(Z1)
    res.append(Z2)
    res.append(Z3)
    res.append(Z4)
    Z1 = circle_normal_model.score(normal)
    Z2 = cross_normal_model.score(normal)
    Z3 = left_slide_normal_model.score(normal)
    Z4 = right_slide_normal_model.score(normal)
    Z5 = unrecogonized_normal_model.score(normal)
    print "normal 模型打分", Z1, Z2, Z3, Z4, Z5

    res.append(Z1)
    res.append(Z2)
    res.append(Z3)
    res.append(Z4)
    Z1 = circle_radius_model.score(radis)
    Z2 = cross_radius_model.score(radis)
    Z3 = left_slide_radius_model.score(radis)
    Z4 = right_slide_radius_model.score(radis)
    Z5 = unrecogonized_radius_model.score(radis)
    print "radius 模型打分", Z1, Z2, Z3, Z4, Z5
    res.append(Z1)
    res.append(Z2)
    res.append(Z3)
    res.append(Z4)
    Z1 = circle_velocity_model.score(velocity)
    Z2 = cross_velocity_model.score(velocity)
    Z3 = left_slide_velocity_model.score(velocity)
    Z4 = right_slide_velocity_model.score(velocity)
    Z5 = unrecogonized_velocity_model.score(velocity)
    print "velocity 模型打分", Z1, Z2, Z3, Z4, Z5
    res.append(Z1)
    res.append(Z2)
    res.append(Z3)
    res.append(Z4)
    normal_res = preprocessing.normalize(res)
    out.append(list(normal_res[0]))
    return out

# 获得神经网络的输入结点的集合
def hmm_score_train_nn(gesture, feature="feature"):
    test_isTurn, lenT = load(gesture + "/isTurn", feature)
    test_normal, lenN = load(gesture + "/normal", feature)
    test_radis, lenR = load(gesture + "/radis", feature)
    test_velocity, lenV = load(gesture + "/velocity", feature)
    out = []
    isTurn = []
    normal = []
    radis = []
    velocity = []
    trainIndex = 0
    trainLen = lenT[trainIndex]
    index = 0
    # 获取神经网路的训练集
    for t, n, r, v in izip(test_isTurn, test_normal, test_radis, test_velocity):
        if index >= trainLen:
            index = 0
            trainIndex += 1
            if trainIndex < len(lenT):
                trainLen = lenT[trainIndex]
            # 计算该组训练集的score
            res = []
            Z1 = circle_isTurn_model.score(isTurn)
            Z2 = cross_isTurn_model.score(isTurn)
            Z3 = left_slide_isTurn_model.score(isTurn)
            Z4 = right_slide_isTurn_model.score(isTurn)
            res.append(Z1)
            res.append(Z2)
            res.append(Z3)
            res.append(Z4)
            Z1 = circle_normal_model.score(normal)
            Z2 = cross_normal_model.score(normal)
            Z3 = left_slide_normal_model.score(normal)
            Z4 = right_slide_normal_model.score(normal)
            res.append(Z1)
            res.append(Z2)
            res.append(Z3)
            res.append(Z4)
            Z1 = circle_radius_model.score(radis)
            Z2 = cross_radius_model.score(radis)
            Z3 = left_slide_radius_model.score(radis)
            Z4 = right_slide_radius_model.score(radis)
            res.append(Z1)
            res.append(Z2)
            res.append(Z3)
            res.append(Z4)
            Z1 = circle_velocity_model.score(velocity)
            Z2 = cross_velocity_model.score(velocity)
            Z3 = left_slide_velocity_model.score(velocity)
            Z4 = right_slide_velocity_model.score(velocity)
            res.append(Z1)
            res.append(Z2)
            res.append(Z3)
            res.append(Z4)
            normal_res = preprocessing.normalize(res)
            out.append(list(normal_res[0]))
            isTurn = []
            normal = []
            radis = []
            velocity = []
        # 拼接每组的训练集结果
        isTurn.append(t)
        normal.append(n)
        radis.append(r)
        velocity.append(v)
        index += 1
    res = []
    Z1 = circle_isTurn_model.score(isTurn)
    Z2 = cross_isTurn_model.score(isTurn)
    Z3 = left_slide_isTurn_model.score(isTurn)
    Z4 = right_slide_isTurn_model.score(isTurn)
    res.append(Z1)
    res.append(Z2)
    res.append(Z3)
    res.append(Z4)
    Z1 = circle_normal_model.score(normal)
    Z2 = cross_normal_model.score(normal)
    Z3 = left_slide_normal_model.score(normal)
    Z4 = right_slide_normal_model.score(normal)
    res.append(Z1)
    res.append(Z2)
    res.append(Z3)
    res.append(Z4)
    Z1 = circle_radius_model.score(radis)
    Z2 = cross_radius_model.score(radis)
    Z3 = left_slide_radius_model.score(radis)
    Z4 = right_slide_radius_model.score(radis)
    res.append(Z1)
    res.append(Z2)
    res.append(Z3)
    res.append(Z4)
    Z1 = circle_velocity_model.score(velocity)
    Z2 = cross_velocity_model.score(velocity)
    Z3 = left_slide_velocity_model.score(velocity)
    Z4 = right_slide_velocity_model.score(velocity)
    res.append(Z1)
    res.append(Z2)
    res.append(Z3)
    res.append(Z4)
    normal_res = preprocessing.normalize(res)
    out.append(list(normal_res[0]))
    return out


h = time.time()

# 创建4个手势4个特征高斯HMM模型，以及一个拒识手势模型
# 左划手势
left_slide_isTurn_model = hmm.GaussianHMM(n_components=4)
left_slide_normal_model = hmm.GaussianHMM(n_components=4)
left_slide_radius_model = hmm.GaussianHMM(n_components=4)
left_slide_velocity_model = hmm.GaussianHMM(n_components=4)
# 导入手势模型数据
isTurn, lensT = load("left_slide/isTurn")
normal, lensN = load("left_slide/normal")
radis, lensR = load("left_slide/radis")
velocity, lensV = load("left_slide/velocity")
# 训练该手势模型
left_slide_isTurn_model.fit(isTurn, lensT)
left_slide_normal_model.fit(normal, lensN)
left_slide_radius_model.fit(radis, lensR)
left_slide_velocity_model.fit(velocity, lensV)

# 右划手势
right_slide_isTurn_model = hmm.GaussianHMM(n_components=4)
right_slide_normal_model = hmm.GaussianHMM(n_components=4)
right_slide_radius_model = hmm.GaussianHMM(n_components=4)
right_slide_velocity_model = hmm.GaussianHMM(n_components=4)
# 导入手势模型数据
isTurn, lensT = load("right_slide/isTurn")
normal, lensN = load("right_slide/normal")
radis, lensR = load("right_slide/radis")
velocity, lensV = load("right_slide/velocity")
# 训练该手势模型
right_slide_isTurn_model.fit(isTurn, lensT)
right_slide_normal_model.fit(normal, lensN)
right_slide_radius_model.fit(radis, lensR)
right_slide_velocity_model.fit(velocity, lensV)

# 画圆手势
circle_isTurn_model = hmm.GaussianHMM(n_components=4)
circle_normal_model = hmm.GaussianHMM(n_components=4)
circle_radius_model = hmm.GaussianHMM(n_components=4)
circle_velocity_model = hmm.GaussianHMM(n_components=4)
# 导入手势模型数据
isTurn, lensT = load("circle/isTurn")
normal, lensN = load("circle/normal")
radis, lensR = load("circle/radis")
velocity, lensV = load("circle/velocity")
# 训练该手势模型
circle_isTurn_model.fit(isTurn, lensT)
circle_normal_model.fit(normal, lensN)
circle_radius_model.fit(radis, lensR)
circle_velocity_model.fit(velocity, lensV)

# 画叉手势
cross_isTurn_model = hmm.GaussianHMM(n_components=4)
cross_normal_model = hmm.GaussianHMM(n_components=4)
cross_radius_model = hmm.GaussianHMM(n_components=4)
cross_velocity_model = hmm.GaussianHMM(n_components=4)
# 导入手势模型数据
isTurn, lensT = load("cross/isTurn")
normal, lensN = load("cross/normal")
radis, lensR = load("cross/radis")
velocity, lensV = load("cross/velocity")
# 训练该手势模型
cross_isTurn_model.fit(isTurn, lensT)
cross_normal_model.fit(normal, lensN)
cross_radius_model.fit(radis, lensR)
cross_velocity_model.fit(velocity, lensV)

# 拒识手势
unrecogonized_isTurn_model = zqx_utils.build_unrecogonized_model([circle_isTurn_model,
                                                                  cross_isTurn_model,
                                                                  left_slide_isTurn_model,
                                                                  right_slide_isTurn_model])
unrecogonized_normal_model = zqx_utils.build_unrecogonized_model([circle_normal_model,
                                                                  cross_normal_model,
                                                                  left_slide_normal_model,
                                                                  right_slide_normal_model])
unrecogonized_radius_model = zqx_utils.build_unrecogonized_model([circle_radius_model,
                                                                  cross_radius_model,
                                                                  left_slide_radius_model,
                                                                  right_slide_radius_model])
unrecogonized_velocity_model = zqx_utils.build_unrecogonized_model([circle_velocity_model,
                                                                    cross_velocity_model,
                                                                    left_slide_velocity_model,
                                                                    right_slide_velocity_model])




print "训练16组HMM模型，耗时：",(time.time()-h),"秒"
nn = time.time()

# 获得训练集输入模型后的似然结果集合
circle_train_samples = hmm_score_train_nn("circle")
cross_train_samples = hmm_score_train_nn("cross")
left_slide_train_samples = hmm_score_train_nn("left_slide")
right_slide_train_samples = hmm_score_train_nn("right_slide")

# 使用训练集集合训练神经网络
mlp = MLPClassifier(hidden_layer_sizes=(256,), learning_rate_init= 0.01)
X = circle_train_samples + cross_train_samples + left_slide_train_samples + right_slide_train_samples
y = [["circle"]] * len(circle_train_samples) + [["cross"]] * len(cross_train_samples) + [["left_slide"]] * len(
    left_slide_train_samples) + [["right_slide"]] * len(right_slide_train_samples)
mlp.fit(X,y)
print "训练16个输入结点，256个隐藏结点的nn模型，耗时：",(time.time()-nn),"秒"
predict = time.time()
print "总共训练HMM-NN模型时间，耗时：",(time.time()-h),"秒"
gt = time.time()
gsc = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
gsc.fit(X,y)
print "训练高斯模型，耗时：",(time.time()-gt),"秒"
print "总共训练HMM-gpc模型时间，耗时：",(time.time()-h)-(gt-nn),"秒"





# 获得测试集输入模型后的似然结果集合
circle_test_samples = hmm_score_train_nn("test", "circle")[0]
cross_test_samples = hmm_score_train_nn("test", "cross")[0]
left_slide_test_samples = hmm_score_train_nn("test", "left_slide")[0]
right_slide_test_samples = hmm_score_train_nn("test", "right_slide")[0]

# 测试拒识模型
print "测试拒识模型：circle"
circle_test_samples1 = hmm_score_get_test_nn("test", "circle")[0]
print "测试拒识模型：cross"
cross_test_samples1 = hmm_score_get_test_nn("test", "cross")[0]
print "测试拒识模型：left_slide"
left_slide_test_samples1 = hmm_score_get_test_nn("test", "left_slide")[0]
print "测试拒识模型right_slide"
right_slide_test_samples1 = hmm_score_get_test_nn("test", "right_slide")[0]

# 使用结果集进行预测
print mlp.predict(circle_test_samples)
print gsc.predict(circle_test_samples)
print gsc.predict(cross_test_samples)
print gsc.predict(left_slide_test_samples)
print gsc.predict(right_slide_test_samples)
print "对测试集进行预测，耗时：",(time.time()-predict),"秒"


#
#
# # different learning rate schedules and momentum parameters
# params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
#            'learning_rate_init': 0.2},
#           {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
#            'nesterovs_momentum': False, 'learning_rate_init': 0.2},
#           {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
#            'nesterovs_momentum': True, 'learning_rate_init': 0.2},
#           {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
#            'learning_rate_init': 0.2},
#           {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
#            'nesterovs_momentum': True, 'learning_rate_init': 0.2},
#           {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
#            'nesterovs_momentum': False, 'learning_rate_init': 0.2},
#           {'solver': 'adam', 'learning_rate_init': 0.01}]
#
# labels = ["constant learning-rate", "constant with momentum",
#           "constant with Nesterov's momentum",
#           "inv-scaling learning-rate", "inv-scaling with momentum",
#           "inv-scaling with Nesterov's momentum", "adam"]
#
# plot_args = [{'c': 'red', 'linestyle': '-'},
#              {'c': 'green', 'linestyle': '-'},
#              {'c': 'blue', 'linestyle': '-'},
#              {'c': 'red', 'linestyle': '--'},
#              {'c': 'green', 'linestyle': '--'},
#              {'c': 'blue', 'linestyle': '--'},
#              {'c': 'black', 'linestyle': '-'}]
#
#
# def plot_on_dataset(X, y, ax, name):
#     # for each dataset, plot learning for each learning strategy
#     print("\nlearning on dataset %s" % name)
#     ax.set_title(name)
#     X = MinMaxScaler().fit_transform(X)
#     mlps = []
#     if name == "digits":
#         # digits is larger but converges fairly quickly
#         max_iter = 15
#     else:
#         max_iter = 200
#
#     for label, param in zip(labels, params):
#         print("training: %s" % label)
#         mlp = MLPClassifier(verbose=0, random_state=0,
#                             max_iter=max_iter, **param)
#         mlp.fit(X, y)
#         mlps.append(mlp)
#         print("Training set score: %f" % mlp.score(X, y))
#         print("Training set loss: %f" % mlp.loss_)
#     for mlp, label, args in zip(mlps, labels, plot_args):
#             ax.plot(mlp.loss_curve_, label=label, **args)
#
#
# fig, axes = plt.subplots(1, 1, figsize=(15, 10))
# # load / generate some toy datasets
# iris = datasets.load_iris()
# digits = datasets.load_digits()
# data_sets = [(iris.data, iris.target),
#              (digits.data, digits.target),
#              datasets.make_circles(noise=0.2, factor=0.5, random_state=1),
#              datasets.make_moons(noise=0.3, random_state=0)]
#
# plot_on_dataset(X,y, ax=axes, name="gesture")
#
# fig.legend(axes.get_lines(), labels=labels, ncol=3, loc="upper center")
# plt.show()