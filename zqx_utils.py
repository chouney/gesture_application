# -*- coding:utf-8 -*-
import numpy as np
import os
from hmmlearn import hmm
import json
import matplotlib.pyplot as plt

# 构造单个通道的拒识模型
def build_unrecogonized_model(models):
    n_components = 0
    transmat = []
    means = []
    covars = []
    for k,v in enumerate(models):
        n_components += v.n_components
        for i in range(v.n_components):
            means.append(v.means_[i])
            # for j in range(v.n_features):
            covars.append(v.covars_[i])
    index = 0
    for k,v in enumerate(models):
        for i in range(v.n_components):
            tmp = []
            aii = v.transmat_[i][i]
            for j in range(n_components):
                aij = (1.0-aii)*1.0/(n_components-1)*1.0
                tmp.append(aij)
            tmp[index] = aii
            transmat.append(tmp)
        index+=1

    start_prob = np.array([1.0/n_components] * n_components)
    covars = np.array(covars)
    covars1 = .5 * np.tile(np.identity(12), (16, 1, 1))
    model = hmm.GaussianHMM(n_components=n_components,covariance_type="full")
    model.startprob_ = np.array(start_prob)
    model.transmat_ = np.array(transmat)
    model.means_ = np.array(means)
    model.covars_ = covars
    model.n_features = models[0].n_features
    return model

# 模型可视化
def draw(X, Z, model):
    # Plot the sampled data
    plt.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6,
             mfc="orange", alpha=0.7)

    # Indicate the component numbers
    for i, m in enumerate(model.means_):
        plt.text(m[0], m[1], 'Component %i' % (i + 1),
                 size=17, horizontalalignment='center',
                 bbox=dict(alpha=.7, facecolor='w'))
    plt.legend(loc='best')
    plt.show()


# #假数据
# test_model = []
# startprob = np.array([0.6, 0.3, 0.1, 0.0])
# # The transition matrix, note that there are no transitions possible
# # between component 1 and 3
# transmat = np.array([[0.7, 0.2, 0.0, 0.1],
#                      [0.3, 0.5, 0.2, 0.0],
#                      [0.0, 0.3, 0.5, 0.2],
#                      [0.2, 0.0, 0.2, 0.6]])
# # The means of each component
# means = np.array([[0.0,  0.0],
#                   [0.0, 11.0],
#                   [9.0, 10.0],
#                   [11.0, -1.0]])
# # The covariance of each component
# covars = .5 * np.tile(np.identity(2), (4, 1, 1))
#
# # Build an HMM instance and set parameters
# model = hmm.GaussianHMM(n_components=4, covariance_type="full")
# model.startprob_ = startprob
# model.transmat_ = transmat
# model.means_ = means
# model.covars_ = covars
# test_model.append(model)
# model = hmm.GaussianHMM(n_components=4, covariance_type="full")
# model.startprob_ = startprob
# model.transmat_ = transmat
# model.means_ = means
# model.covars_ = covars
# test_model.append(model)
# model = hmm.GaussianHMM(n_components=4, covariance_type="full")
# model.startprob_ = startprob
# model.transmat_ = transmat
# model.means_ = means
# model.covars_ = covars
# test_model.append(model)
# model = hmm.GaussianHMM(n_components=4, covariance_type="full")
# model.startprob_ = startprob
# model.transmat_ = transmat
# model.means_ = means
# model.covars_ = covars
# test_model.append(model)
#
#
#
# # 拒识手势
# trans_prob = np.array([[0.0625]*16]*16)
# unrecogonized_isTurn_model = hmm.GaussianHMM(n_components=16)
# unrecogonized_normal_model = hmm.GaussianHMM(n_components=16)
# unrecogonized_radius_model = hmm.GaussianHMM(n_components=16)
# unrecogonized_velocity_model = hmm.GaussianHMM(n_components=16)
# unrecogonized_test_model = build_unrecogonized_model(test_model)
# X, Z = unrecogonized_test_model.sample(500)
# draw(X, Z, unrecogonized_test_model)
# X, Z = unrecogonized_isTurn_model.sample(500)
# draw(X, Z)
# X, Z = unrecogonized_normal_model.sample(500)
# draw(X, Z)
# X, Z = unrecogonized_radius_model.sample(500)
# draw(X, Z)
# X, Z = unrecogonized_velocity_model.sample(500)
# draw(X, Z)

