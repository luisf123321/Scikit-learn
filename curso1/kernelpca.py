import pandas as pd 
import sklearn 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA
from sklearn.linear_model  import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    dt_hear = pd.read_csv("data/heart.csv")
    print(dt_hear.head(5))
    dt_feature = dt_hear.drop(['target'],axis=1)
    dt_target = dt_hear['target']
    dyt_feature = StandardScaler().fit_transform(dt_feature)

    x_train, x_test, y_train, y_test = train_test_split(dt_feature,dt_target,test_size=0.3,random_state=42)

    print(x_train.shape)
    print(y_train.shape)

    kpca = KernelPCA(n_components=4,kernel='poly')

    kpca.fit(x_train)

    dt_train = kpca.transform(x_train)
    dt_test = kpca.transform(x_test)

    log = LogisticRegression(solver='lbfgs')
    log.fit(dt_train,y_train)

    print("score kernel kpca ", log.score(dt_test,y_test))





