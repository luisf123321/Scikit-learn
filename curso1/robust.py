import pandas as pd 
import sklearn 
import matplotlib
from sklearn import model_selection
matplotlib.use('TkAgg')


from sklearn.linear_model import (
    RANSACRegressor,
    HuberRegressor
)

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv("./data/felicidad_corrupt.csv")

    print(dataset.head(5))

    X = dataset.drop(['score','country'],axis=1)
    y = dataset[['score']]

    x_train, x_test, y_train, y_test  = train_test_split(X,y,test_size=0.3,random_state=42)

    estimadores = {
        'SVR': SVR(gamma="auto",C=1.0,epsilon=0.1),
        "RANSAC": RANSACRegressor(),
        "HUBER": HuberRegressor(epsilon=1.35)
    }

    for name, estimador  in estimadores.items():
        estimador.fit(x_train,y_train)
        preditions = estimador.predict(x_test)
        print("="*32)
        print(name)
        print("MSE",mean_squared_error(y_test,preditions))
