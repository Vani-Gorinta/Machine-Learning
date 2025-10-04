import pandas as pd
iris = pd.read_csv(r"C:\Users\vanig\Downloads\IRIS.csv")

x = iris.iloc[:,: -1].values
y = iris.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3)
y_pred = hc.fit_predict(x)

import matplotlib.pyplot as plt

plt.scatter(x[kmeans == 0,0], x[kmeans == 0,1], s=50, c='blue', label='Setosa')
plt.scatter(x[kmeans == 1,0], x[kmeans == 1,1], s=50, c='red', label='Versicolor')
plt.scatter(x[kmeans == 2,0], x[kmeans == 2,1], s=50, c='green', label='Verginica')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=150, c='black', label='centroids')

from scipy.cluster.hierarchy import dendrogram, linkage
dendrogram(linkage(x))

plt.legend
plt.show()