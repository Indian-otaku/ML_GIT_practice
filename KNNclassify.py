import numpy as np
import warnings
import pandas as pd
from collections import Counter

# Branch to test out whether the KNN classifier works well using real data and compared it with sklearn kNN classifier

def KNNclassifier(data, predict, k=3):
    if (len(data) >= k):
        warnings.warn('Cannot have equal to or more classes than value of k!')

    distances = []
    for classes in data:
        for features in data[classes]:
            dist = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([dist, classes])

    vote = [i[1] for i in sorted(distances) [:k]]
    vote_result = Counter(vote).most_common(1)[0][0]

    return vote_result

df = pd.read_csv('breast-cancer-wisconsin.data',na_values='?')
df.replace(np.nan,-99999,inplace=True)
print(df.head(25)['Bare_Nuclei'])