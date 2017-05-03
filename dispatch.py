# -*- coding:utf-8 -*-
import numpy as np
import os
from hmmlearn import hmm
import json
import matplotlib.pyplot as plt

np.random.seed(42)


def store(data, filepackname, filename):
    if os.path.exists(filepackname)== False:
        os.makedirs(filepackname)
    with open(filepackname+'/'+filename, 'w') as json_file:
        json_file.write(json.dumps(data))


def load(filepack,prefilename="feature"):
    datas = []
    files = os.listdir(filepack)
    for name in files:
        if name.startswith(prefilename):
            with open(filepack+"/"+name) as json_file:
                data = json.load(json_file)["cur"]
                normal = []
                velocity = []
                radis = []
                isTurn = []
                for row in data:
                    normal.append(row[0:3])
                    velocity.append(row[3:6])
                    radis.append([row[6]])
                    isTurn.append(row[7:12])
                store(normal, filepack + "/normal", name)
                store(velocity, filepack + "/velocity", name)
                store(radis, filepack + "/radis", name)
                store(isTurn, filepack + "/isTurn", name)

    return datas

a = [[1],[2],[3],[4],[5]]
print a[1:2]
