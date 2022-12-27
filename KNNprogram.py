import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

accuracies = []
for i in range(20):
    df = pd.read_csv('breast-cancer-wisconsin.data',na_values='?')
    df.replace(np.nan,-99999,inplace=True)
    df.drop(['Sample_code_number'],axis='columns',inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(df.drop(['Class'],axis='columns'),df['Class'],test_size=0.2)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train,y_train)
    accuracy = model.score(X_test,y_test)
    accuracies.append(accuracy)

print("Average accuracy: ",sum(accuracies)/len(accuracies))