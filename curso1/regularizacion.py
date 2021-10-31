import pandas as pd 
import sklearn 
import matplotlib
from sklearn import model_selection
matplotlib.use('TkAgg')

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset.describe())

    x = dataset[['gdp','family','lifexp','freedom','corruption','generosity','dystopia']]
    y = dataset[['score']]

    print(x.shape)
    print(y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)

    modelLineal = LinearRegression().fit(x_train,y_train)
    y_prediction = modelLineal.predict(x_test)


    modelLasso = Lasso(alpha=0.02,).fit(x_train,y_train)
    y_prediction_lasso = modelLasso.predict(x_test)

    modelRidge = Ridge(alpha=1).fit(x_train,y_train)
    y_prediction_ridge = modelRidge.predict(x_test)

    lineal_loss = mean_squared_error(y_test,y_prediction)
    print("lineal loss",lineal_loss)

    lasso_loss = mean_squared_error(y_test,y_prediction_lasso)
    print("lineal lasso loss",lasso_loss)

    ridge_loss = mean_squared_error(y_test,y_prediction_ridge)
    print("ridge loss",ridge_loss)

    print("="*32)
    print("coeficiente lasso")
    print(modelLasso.coef_)

    print("coeficiente Ridge")
    print(modelRidge.coef_)



