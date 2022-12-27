import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

data = {'r':[[1,2],[2,4],[1,3],[1,4]],'b':[[6,7],[8,6],[7,9],[5,8]]}

def KNNclassifier(data, predict, k=3):
    if (len(data) >= k):
        warnings.warn('Cannot have equal to or more classes than value of k!')

    distances = []
    for classes in data:
        for features in data[classes]:
            dist = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append(dist, classes)

    vote_result = Counter(distances).most_common(1)[0][0]

    return vote_result

[[plt.scatter(ii[0],ii[1],s=100,c=i) for ii in data[i]] for i in data]
predict = [3,5]
predict_class = KNNclassifier(data,predict,k=3)
plt.scatter(*predict,s=150,c=predict_class)
plt.show()