import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.cluster import KMeans

dataset = pd.read_csv('./Customers.csv')

X = dataset.iloc[:, [2,6]].values

wcss = []

for i in range(1, 100):
    kmeans = KMeans(n_clusters = i, init='k-means++')
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plot.figure(figsize=(10, 5))
sns.set(context='paper', palette='Blues', style='whitegrid', font_scale=1, color_codes=True)
sns.lineplot(x=range(1, 100), y=wcss, markers='o', color='red')
plot.show()

#The Saturation Point is at 20
#After Seeing the saturation point close the window
#So K value is 20
#So we are having 20 groups

kmeans = KMeans(n_clusters=10, init='k-means++')
y_means = kmeans.fit_predict(X)

plot.figure(figsize=(12, 7))
sns.scatterplot(x=X[y_means == 0, 0], y=X[y_means == 0, 1], color='yellow', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 1, 0], y=X[y_means == 1, 1], color='blue', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 2, 0], y=X[y_means == 2, 1], color='green', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 3, 0], y=X[y_means == 3, 1], color='orange', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 4, 0], y=X[y_means == 4, 1], color='grey', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 5, 0], y=X[y_means == 5, 1], color='yellow', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 6, 0], y=X[y_means == 6, 1], color='blue', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 7, 0], y=X[y_means == 7, 1], color='green', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 8, 0], y=X[y_means == 8, 1], color='orange', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 9, 0], y=X[y_means == 9, 1], color='grey', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 10, 0], y=X[y_means == 10, 1], color='yellow', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 11, 0], y=X[y_means == 11, 1], color='blue', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 12, 0], y=X[y_means == 12, 1], color='green', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 13, 0], y=X[y_means == 13, 1], color='orange', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 14, 0], y=X[y_means == 14, 1], color='grey', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 15, 0], y=X[y_means == 15, 1], color='yellow', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 16, 0], y=X[y_means == 16, 1], color='blue', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 17, 0], y=X[y_means == 17, 1], color='green', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 18, 0], y=X[y_means == 18, 1], color='orange', label='Cluster 1', s=50)
sns.scatterplot(x=X[y_means == 19, 0], y=X[y_means == 19, 1], color='grey', label='Cluster 1', s=50)


sns.scatterplot(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:,1], color='red', label='centroids', s=200, marker=',')
plot.grid(False)
plot.title("Cluster of Customers")
plot.xlabel("Annual Income")
plot.ylabel("Family Size")
plot.legend()
plot.show()