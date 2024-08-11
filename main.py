import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import zscore

data = pd.read_csv("CC GENERAL.csv")

data = data.drop(columns="CUST_ID")
data["MINIMUM_PAYMENTS"].fillna(data["MINIMUM_PAYMENTS"].mean(), inplace=True)
data["CREDIT_LIMIT"].fillna(data["CREDIT_LIMIT"].mean(), inplace=True)

data_scaled = data.apply(zscore)

optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
kmeans.fit(data_scaled)

print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)

labels = kmeans.predict(data_scaled)

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

pca_df = pd.DataFrame(data_pca)
pca_df['cluster'] = labels

def kmeans_display(X, labels):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', s=60, alpha=0.7)
    plt.title('KMeans Clustering Results')
    plt.legend(title='Cluster')
    plt.show()

kmeans_display(data_pca, labels)
