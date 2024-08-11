import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import zscore
data = pd.read_csv("CC GENERAL.csv").drop(columns="CUST_ID")
data.fillna(data.mean(), inplace=True)
data_scaled = data.apply(zscore)
kmeans = KMeans(n_clusters=4, random_state=0).fit(data_scaled)
labels = kmeans.labels_
data_pca = PCA(n_components=2).fit_transform(data_scaled)
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=labels, palette='viridis', s=60, alpha=0.7)
plt.title('KMeans Clustering Results')
plt.legend(title='Cluster')
plt.show()