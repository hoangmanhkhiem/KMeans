import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import zscore
data = pd.read_csv("CC GENERAL.csv").drop(columns="CUST_ID")
data_scaled = data.apply(lambda x: zscore(x.fillna(x.mean())))
labels = KMeans(n_clusters=4, random_state=0).fit_predict(data_scaled)
data_pca = PCA(n_components=2).fit_transform(data_scaled)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', s=60, alpha=0.7)
plt.show()
