import pandas as pd
from scipy.sparse import data 
import sklearn 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    dataset = pd.read_csv("./data/heart.csv")
    print(dataset['target'].describe())

    X = dataset.drop(['target'],axis=1)
    y = dataset['target']

    x_train, x_test, y_train, y_test  =  train_test_split(X,y,test_size=0.35)

    knn_class = KNeighborsClassifier().fit(x_train,y_train)
    knn_preditions = knn_class.predict(x_test)

    print("="*34)
    print(accuracy_score(knn_preditions,y_test))


    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(),n_estimators=50).fit(x_train,y_train)
    bag_preditions = bag_class.predict(x_test)

    print("="*34)
    print(accuracy_score(bag_preditions,y_test))