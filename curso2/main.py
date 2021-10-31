from utils import Utils
from model import Models

if __name__ == "__main__":

    utils = Utils()
    models = Models()

    data = utils.load_from_csv('./in/felicidad.csv')

    x, y = utils.features_target(data,['score','rank','country'],['score'])

    models.grid_training(x,y)

    print(data)
