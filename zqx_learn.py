# -*- coding:utf-8 -*-
import numpy as np
import os
from hmmlearn import hmm
import json
import matplotlib.pyplot as plt
from sklearn import preprocessing
import warnings
import time
import zqx_utils
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
                data = json.load(json_file)['cur']
                for row in data:
                    datas.append(row)
                    sample += 1
            lens.append(sample)
    return datas, lens


def test_score(gesture):
    test_list, test_len = load(gesture, start=80)
    X = []
    train_index = 0
    index = 0
    trainlen = test_len[train_index]
    loss_num = 0
    for r in test_list:
        if index >= trainlen:
            map = {
                'left_slide':left_slide_model.score(X),
                'right_slide':right_slide_model.score(X),
                'circle':circle_model.score(X),
                'cross':cross_model.score(X),
                'uncogonized':uncogonized_model.score(X)
            }

            m_gesture = max(zip(map.values(), map.keys()))[1]
            if m_gesture == 'uncogonized':
                print "手势：", gesture, "被拒识,detail", X
            if m_gesture != gesture:
                loss_num += 1
            print "测试手势：",gesture, "识别结果：", m_gesture
            train_index+=1
            trainlen = test_len[train_index]
            index = 0
            X = []
        index += 1
        X.append(r)
    map = {
        'left_slide': left_slide_model.score(X),
        'right_slide': right_slide_model.score(X),
        'circle': circle_model.score(X),
        'cross': cross_model.score(X),
        'uncogonized': uncogonized_model.score(X)
    }
    m_gesture = max(zip(map.values(), map.keys())[1])
    if m_gesture == 'uncogonized':
        print "手势：", gesture, "被拒识,detail", X
    if m_gesture != gesture:
        loss_num += 1
    print "测试手势：", gesture, "识别结果：", m_gesture
    return loss_num, len(test_len)

# 创建一个高斯HMM模型
n = time.time()
left_slide_model = hmm.GaussianHMM(n_components=4)
right_slide_model= hmm.GaussianHMM(n_components=4)
circle_model= hmm.GaussianHMM(n_components=4)
cross_model= hmm.GaussianHMM(n_components=4)

left_slide, lslen = load("./left_slide", end=80)
right_slide, rslen = load("./right_slide", end=80)
circle, clen = load("./circle", end=80)
cross, crlen = load("./cross", end=80)
# X = np.concatenate([left_slide, right_slide, circle, cross])
# lengths = [len(left_slide),len(right_slide),len(circle),len(cross)]
# model.fit(X, lengths)
left_slide_model.fit(left_slide, lslen)
right_slide_model.fit(right_slide, rslen)
circle_model.fit(circle, clen)
cross_model.fit(cross, crlen)
# 创建拒识模型
uncogonized_model = zqx_utils.build_unrecogonized_model([circle_model, cross_model,left_slide_model, right_slide_model])

print "训练4组HMM模型，耗时：",time.time()-n," 秒"

loss = 0
total_len = 0
print "测试画圆手势模型："
loss_num, clen = test_score("circle")
loss+=loss_num
total_len += clen
print "测试画叉手势模型："
loss_num, clen = test_score("cross")
loss+=loss_num
total_len += clen
print "测试左划手势模型："
loss_num, clen = test_score("left_slide")
loss+=loss_num
total_len += clen
print "测试右划手势模型："
loss_num, clen = test_score("right_slide")
loss+=loss_num
total_len += clen
print "测试正确率：",1.0-loss*1.0/total_len
# zqx_utils.draw(X, Z, uncogonized_model)
