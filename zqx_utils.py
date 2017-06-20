# -*- coding:utf-8 -*-
import numpy as np
import os
from hmmlearn import hmm
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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


# 给定数据的分类器可视化
def classfication_loss(X_train , y_train, X_test, y_test):
    h = .02  # step size in the mesh

    # names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
    #          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
    #          "Naive Bayes", "QDA"]

    names = ["Decision Tree"]

    classifiers = [
        # KNeighborsClassifier(3),
        # SVC(kernel="linear", C=0.025),
        # SVC(gamma=2, C=1),
        # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        DecisionTreeClassifier(max_depth=5),
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # MLPClassifier(alpha=1),
        # AdaBoostClassifier(),
        # GaussianNB(),
        # QuadraticDiscriminantAnalysis()]
    ]
    for name, clf in zip(names, classifiers):
        start = time.time()
        clf.fit(X_train,y_train)
        print "分类器：",name,"训练用时：", time.time()-start
        start = time.time()
        Z = clf.predict(X_test)
        test = np.array(y_test)[:, 0]
        print "预测用时：", time.time()-start
        loss = 0
        for i in range(0,len(Z)):
            if Z[i] != test[i]:
                loss += 1
        print "准确率：",1.0-(loss*1.0/len(Z)), "detail：", [Z[i] + ":" + test[i] for i in range(0, len(Z))]