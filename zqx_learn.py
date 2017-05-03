# -*- coding:utf-8 -*-
import numpy as np
import os
from hmmlearn import hmm
import json
import matplotlib.pyplot as plt
import warnings
import time
import zqx_utils
warnings.filterwarnings("ignore")
np.random.seed(42)


def store(data):
    with open('feature', 'w') as json_file:
        json_file.write(json.dumps(data))


def load(filepack,prefilename="feature"):
    datas = []
    lens = []
    files = os.listdir(filepack)
    for name in files:
        if name.startswith(prefilename):
            sample = 0
            with open(filepack+"/"+name) as json_file:
                data = json.load(json_file)["cur"]
                for row in data:
                    datas.append(row)
                    sample+=1
            lens.append(sample)
    return datas, lens


# 创建一个高斯HMM模型
n = time.time()
left_slide_model = hmm.GaussianHMM(n_components=4)
right_slide_model= hmm.GaussianHMM(n_components=4)
circle_model= hmm.GaussianHMM(n_components=4)
cross_model= hmm.GaussianHMM(n_components=4)

left_slide, lslen = load("./left_slide")
right_slide, rslen = load("./right_slide")
circle, clen = load("./circle")
cross, crlen = load("./cross")
# X = np.concatenate([left_slide, right_slide, circle, cross])
# lengths = [len(left_slide),len(right_slide),len(circle),len(cross)]
# model.fit(X, lengths)
left_slide_model.fit(left_slide, lslen)
right_slide_model.fit(right_slide, rslen)
circle_model.fit(circle, clen)
cross_model.fit(cross, crlen)
# 创建拒识模型
uncogonized_model = zqx_utils.build_unrecogonized_model([circle_model, cross_model,left_slide_model, right_slide_model])

print "训练16组HMM模型，耗时：",time.time()-n," 秒"

test_circle,lenc = load("./test", "circle")
test_cross, lencr = load("./test", "cross")
test_left_slide,lenle = load("./test", "left_slide")
test_right_slide,lenrs = load("./test", "right_slide")

# Z1 = cross_model.predict_proba(test_circle,[len(test_circle)])
# Z2 = cross_model.predict_proba(test_cross,[len(test_cross)])
# Z3 = cross_model.predict_proba(test_left_slide,[len(test_left_slide)])
# Z4 = cross_model.predict_proba(test_right_slide,[len(test_right_slide)])
Z1 = circle_model.score(test_circle)
Z2 = cross_model.score(test_circle)
Z3 = left_slide_model.score(test_circle)
Z4 = right_slide_model.score(test_circle)
Z5 = uncogonized_model.score(test_circle)
print "测试画圆手势模型："
print Z1
print Z2
print Z3
print Z4
print Z5
Z1 = circle_model.score(test_cross)
Z2 = cross_model.score(test_cross)
Z3 = left_slide_model.score(test_cross)
Z4 = right_slide_model.score(test_cross)
Z5 = uncogonized_model.score(test_cross)

print "测试画叉手势模型："
print Z1
print Z2
print Z3
print Z4
print Z5

Z1 = circle_model.score(test_left_slide)
Z2 = cross_model.score(test_left_slide)
Z3 = left_slide_model.score(test_left_slide)
Z4 = right_slide_model.score(test_left_slide)
Z5 = uncogonized_model.score(test_left_slide)

print "测试左划手势模型："
print Z1
print Z2
print Z3
print Z4
print Z5

Z1 = circle_model.score(test_right_slide)
Z2 = cross_model.score(test_right_slide)
Z3 = left_slide_model.score(test_right_slide)
Z4 = right_slide_model.score(test_right_slide)
Z5 = uncogonized_model.score(test_right_slide)

print "测试右划手势模型："
print Z1
print Z2
print Z3
print Z4
print Z5

X = uncogonized_model.sample(10)[0];
Z1 = circle_model.score(X)
Z2 = cross_model.score(X)
Z3 = left_slide_model.score(X)
Z4 = right_slide_model.score(X)
Z5 = uncogonized_model.score(X)
print "测试拒识手势模型："
print Z1
print Z2
print Z3
print Z4
print Z5
# zqx_utils.draw(X, Z, uncogonized_model)
