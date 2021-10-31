import pandas as pd

from sklearn.cluster import MiniBatchKMeans


if __name__ == "__main__":
    dataset = pd.read_csv("./data/candy.csv")
    print(dataset.head(10))

    X = dataset.drop(['competitorname'],axis=1)

    kmean = MiniBatchKMeans(n_clusters=4,batch_size=8).fit(X)
    print("="*64)
    print("total centros ", len(kmean.cluster_centers_))
    print(kmean.predict(X))

    dataset['group'] = kmean.predict(X)
    print(dataset)