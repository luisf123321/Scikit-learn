import pandas as pd
from scipy.sparse import data 
import sklearn 

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    dataset = pd.read_csv("./data/heart.csv")
    print(dataset['target'].describe())

    X = dataset.drop(['target'],axis=1)
    y = dataset['target']

    x_train, x_test, y_train, y_test  =  train_test_split(X,y,test_size=0.35)

    boot = GradientBoostingClassifier(n_estimators=50).fit(x_train,y_train)
    boot_pred = boot.predict(x_test)

    print("="*64)
    print(accuracy_score(boot_pred,y_test))

    