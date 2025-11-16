import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

matplotlib.use("Agg")
df = pd.read_csv("titanic_data.csv")

# Section 3: K-Means 
df['Age_norm'] = ((df['Age'] - df['Age'].min())/(df['Age'].max()-df['Age'].min())).round(1)
cluster_features = ['Age_norm','Gender_enc','Pclass','Embarked_enc','TravelingAlone']

# Initial centers: PassengerID 4 and 46
c1 = df.loc[df['PassengerID']==4, cluster_features].values[0]
c2 = df.loc[df['PassengerID']==46, cluster_features].values[0]

dist_c1 = np.linalg.norm(df[cluster_features].values - c1, axis=1)
dist_c2 = np.linalg.norm(df[cluster_features].values - c2, axis=1)

df['Cluster'] = np.where(dist_c1<=dist_c2,1,2)

cluster_99 = df.loc[df['PassengerID']==99,'Cluster'].iloc[0]
dist_9_c2 = dist_c2[df[df['PassengerID']==9].index[0]]
cluster_counts = df['Cluster'].value_counts()

print("\n Section 3: K-Means ")
print("Passenger 99 cluster:", cluster_99)
print("Distance Passenger 9 to C2:", round(dist_9_c2,1))
print("Cluster counts:\n", cluster_counts)


pca = PCA(n_components=2)
X_reduced = pca.fit_transform(df[cluster_features])

plt.figure(figsize=(7,6))
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=df['Cluster'], cmap='Set1', alpha=0.6)
plt.title("K-Means Clusters (K=2) — PCA Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("eda_plots.png")
