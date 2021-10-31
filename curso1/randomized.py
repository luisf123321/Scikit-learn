import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    dataset = pd.read_csv("./data/felicidad.csv")

    

    X = dataset.drop(['country','rank','score'],axis=1)
    y = dataset[['score']]

    reg = RandomForestRegressor()
    parametros = {
        'n_estimators':range(4,16),
        'criterion':['mse','mae'],
        'max_depth':range(2,11)

    }

    rand_set = RandomizedSearchCV(reg,parametros,n_iter=10,cv=3,scoring='neg_mean_absolute_error').fit(X,y)

    print(rand_set.best_estimator_)
    print(rand_set.best_params_)

    print(rand_set.predict(X.loc[[0]]))