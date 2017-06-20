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
from sklearn.model_selection import train_test_split
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


def load(filepack, prefilename="feature", start=0, end=-1):
    datas = []
    lens = []
    files = os.listdir(filepack)
    index = -1
    for name in files:
        if name.startswith(prefilename):
            index += 1
            if index < start or (end != -1 and index >= end):
                continue
            sample = 0
            with open(filepack + "/" + name) as json_file:
                data = json.load(json_file)
                for row in data:
                    datas.append(row)
                    sample += 1
            lens.append(sample)
    if filepack.endswith("radis"):
        return preprocessing.MinMaxScaler().fit_transform(datas), lens
    return preprocessing.normalize(datas), lens

# HMM模型打分的私有方法,如果不符合则返回False
def test_and_score(X_isTurn,X_normal,X_radius,X_velocity ,gesture):
    channels = [[X_isTurn,[circle_isTurn_model, cross_isTurn_model,left_slide_isTurn_model,right_slide_isTurn_model,unrecogonized_isTurn_model]],
            [X_normal,[circle_normal_model, cross_normal_model, left_slide_normal_model, right_slide_normal_model, unrecogonized_normal_model]],
            [X_radius,[circle_radius_model, cross_radius_model, left_slide_radius_model, right_slide_radius_model, unrecogonized_radius_model]],
            [X_velocity,[circle_velocity_model, cross_velocity_model, left_slide_velocity_model, right_slide_velocity_model, unrecogonized_velocity_model]]]
    res = []
    unrecognized_count = 0
    for X, models in channels:
        map_score = {
            'circle': 0,
            'cross': 0,
            'left_slide': 0,
            'right_slide': 0,
            'unrecogonized': 0
        }
        keys = map_score.keys()
        for i in range(0, len(models)):
            s = models[i].score(X)
            map_score[keys[i]] = s
            if i != 4:
                res.append(s)
        if max(zip(map_score.values(), map_score.keys()))[1] == 'unrecogonized':
            unrecognized_count += 1
    if unrecognized_count == 4:
        print "手势：", gesture, "被拒识,detail",X_velocity
        # return False
    return res

def score(X_isTurn,X_normal,X_radius,X_velocity):
    channels = [[X_isTurn, [circle_isTurn_model, cross_isTurn_model,left_slide_isTurn_model,right_slide_isTurn_model]],
            [X_normal, [circle_normal_model, cross_normal_model, left_slide_normal_model, right_slide_normal_model]],
            [X_radius, [circle_radius_model, cross_radius_model, left_slide_radius_model, right_slide_radius_model]],
            [X_velocity, [circle_velocity_model, cross_velocity_model, left_slide_velocity_model,
                          right_slide_velocity_model]]]
    res = []
    for X, models in channels:
        for i in range(0, len(models)):
            s = models[i].score(X)
            res.append(s)
    return res

# 获得神经网络的输入结点的集合，如果手势被拒识模型识别，则插入-1
def hmm_score_get_test_nn(gesture, feature="feature"):
    test_isTurn, lenT = load(gesture + "/isTurn", feature, start=80)
    test_normal, lenN = load(gesture + "/normal", feature, start=80)
    test_radis, lenR = load(gesture + "/radis", feature, start=80)
    test_velocity, lenV = load(gesture + "/velocity", feature, start=80)
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
            res = test_and_score(isTurn, normal, radis, velocity, gesture)
            # 被拒识了则不进行输入
            if res != False:
                normal_res = preprocessing.normalize(res)
                out.append(list(normal_res[0]))
            isTurn = []
            normal = []
            radis = []
            velocity = []
        # 拼接每组的训练集结果
        index += 1
        isTurn.append(t)
        normal.append(n)
        radis.append(r)
        velocity.append(v)
    res = test_and_score(isTurn, normal, radis, velocity, gesture)
    if res != False:
        normal_res = preprocessing.normalize(res)
        out.append(list(normal_res[0]))
    return out

# 获得神经网络的输入结点的集合
def hmm_score_train_nn(gesture, feature="feature"):
    test_isTurn, lenT = load(gesture + "/isTurn", feature, start=0, end=80)
    test_normal, lenN = load(gesture + "/normal", feature, start=0, end=80)
    test_radis, lenR = load(gesture + "/radis", feature, start=0, end=80)
    test_velocity, lenV = load(gesture + "/velocity", feature, start=0, end=80)
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
            res = score(isTurn,normal,radis,velocity)
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
    res = score(isTurn,normal,radis,velocity)
    normal_res = preprocessing.normalize(res)
    out.append(list(normal_res[0]))
    return out

# 构建模型手势
def buildGaussianHMMModels(gesture):
    # 模型手势
    isTurn_model = hmm.GaussianHMM(n_components=4)
    normal_model = hmm.GaussianHMM(n_components=4)
    radius_model = hmm.GaussianHMM(n_components=4)
    velocity_model = hmm.GaussianHMM(n_components=4)
    # 导入手势模型数据
    isTurn, lensT = load(gesture+"/isTurn", start=0, end=80)
    normal, lensN = load(gesture+"/normal", start=0, end=80)
    radis, lensR = load(gesture+"/radis", start=0, end=80)
    velocity, lensV = load(gesture+"/velocity", start=0, end=80)
    # 训练该手势模型
    isTurn_model.fit(isTurn, lensT)
    normal_model.fit(normal, lensN)
    radius_model.fit(radis, lensR)
    velocity_model.fit(velocity, lensV)
    return isTurn_model, normal_model, radius_model, velocity_model


h = time.time()

# 创建4个手势4个特征高斯HMM模型，以及一个拒识手势模型
# 左划手势
left_slide_isTurn_model, left_slide_normal_model,\
    left_slide_radius_model, left_slide_velocity_model = buildGaussianHMMModels("left_slide")
# 右划手势
right_slide_isTurn_model, right_slide_normal_model, \
    right_slide_radius_model, right_slide_velocity_model = buildGaussianHMMModels("right_slide")
# 画圆手势
circle_isTurn_model, circle_normal_model, \
    circle_radius_model, circle_velocity_model = buildGaussianHMMModels("circle")
# 画叉手势
cross_isTurn_model, cross_normal_model, \
    cross_radius_model, cross_velocity_model = buildGaussianHMMModels("cross")



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

X_train = circle_train_samples + cross_train_samples + left_slide_train_samples + right_slide_train_samples
y_train = [["circle"]] * len(circle_train_samples) + [["cross"]] * len(cross_train_samples) + [["left_slide"]] * len(
     left_slide_train_samples) + [["right_slide"]] * len(right_slide_train_samples)

print "训练16个输入结点，256个隐藏结点的nn模型，耗时：",(time.time()-nn),"秒"
predict = time.time()
print "总共训练HMM-NN模型时间，耗时：",(time.time()-h),"秒"
# gt = time.time()
# gsc = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
# gsc.fit(X,y)
# print "训练高斯过程模型，耗时：",(time.time()-gt),"秒"
# print "总共训练HMM-gpc模型时间，耗时：",(time.time()-h)-(gt-nn),"秒"

# todo 分类器数据可视化
# zqx_utils.classfication_visualization([(X,y)])




# 测试拒识模型并或获得测试数据
print "测试数据：circle"
circle_test_samples = hmm_score_get_test_nn("circle")
print "测试数据：cross"
cross_test_samples = hmm_score_get_test_nn("cross")
print "测试数据：left_slide"
left_slide_test_samples = hmm_score_get_test_nn("left_slide")
print "测试数据right_slide"
right_slide_test_samples = hmm_score_get_test_nn("right_slide")

# 测试画圆手势准确率
print "测试画圆手势准确率"
X_test = circle_test_samples
y_test = [["circle"]] * len(circle_test_samples)
zqx_utils.classfication_loss(X_train ,y_train, X_test, y_test)

# 测试画叉手势准确率
print "测试画叉手势准确率"
X_test = cross_test_samples
y_test = [["cross"]] * len(cross_test_samples)
zqx_utils.classfication_loss(X_train ,y_train, X_test, y_test)

# 测试左滑手势准确率
print "测试左滑手势准确率"
X_test = left_slide_test_samples
y_test = [["left_slide"]] * len(left_slide_test_samples)
zqx_utils.classfication_loss(X_train ,y_train, X_test, y_test)

# 测试右滑手势准确率
print "测试右滑手势准确率"
X_test = right_slide_test_samples
y_test = [["right_slide"]] * len(right_slide_test_samples)
zqx_utils.classfication_loss(X_train ,y_train, X_test, y_test)

#测试整体手势准确率
print "测试整体手势准确率"
X_test = circle_test_samples + cross_test_samples + left_slide_test_samples + right_slide_test_samples
y_test = [["circle"]] * len(circle_test_samples) + [["cross"]] * len(cross_test_samples) + [["left_slide"]] * len(
    left_slide_test_samples) + [["right_slide"]] * len(right_slide_test_samples)

# 测试各个分类器性能和loss
zqx_utils.classfication_loss(X_train ,y_train, X_test, y_test)

