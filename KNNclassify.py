import random
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

accuracies = []
for i in range(20):
    df = pd.read_csv('breast-cancer-wisconsin.data',na_values='?')
    df.replace(np.nan,-99999,inplace=True)
    df.drop(['Sample_code_number'],axis='columns',inplace=True)

    data_list = df.astype(float).values.tolist()
    random.shuffle(data_list)

    test_size = 0.2
    train_set = {2:[],4:[]}
    test_set = {2:[],4:[]}
    train_data = data_list[:-int(len(data_list)*test_size)]
    test_data = data_list[-int(len(data_list)*test_size):]

    [train_set[i[-1]].append(i[:-1]) for i in train_data]
    [test_set[i[-1]].append(i[:-1]) for i in test_data]

    correct = 0
    total = 0
    for group in test_set:
        for feature in test_set[group]:
            prediction = KNNclassifier(train_set,feature,k=5)
            if (group == prediction):
                correct += 1
            total += 1

    # print("Accuracy: ",correct/total)
    accuracies.append(correct/total)

print("Average accuracy: ",sum(accuracies)/len(accuracies))