import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

matplotlib.use("Agg")
df = pd.read_csv("titanic_data.csv")

# Section 2: K-NN
features = ['Age','Gender_enc','Pclass','Embarked_enc','TravelingAlone']
X = df[features].values
y = df['Survived'].values

# New passenger
new_pass = np.array([[61,0,2,2,1]])   # Male=0, S=2

# Distances
dists = np.linalg.norm(X - new_pass, axis=1)
df['Distance'] = dists
nearest3 = df.nsmallest(3,'Distance')[['PassengerID','Distance','Survived']]

print("\n Section 2: K-NN ")
print("Top 3 nearest neighbors:\n", nearest3)

# K=5 prediction
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X,y)
pred5 = knn5.predict(new_pass)[0]
print("K=5 prediction:", pred5)

# K=9 neighbors
dist_order = np.argsort(dists)[:9]
nine_neighbors = df.iloc[dist_order]
num_survived_in9 = nine_neighbors['Survived'].sum()
print("K=9 -> Survived=1 among 9 neighbors:", num_survived_in9)

plt.figure(figsize=(6,5))
plt.scatter(df['Age'], df['Pclass'], c=df['Survived'], cmap='coolwarm', alpha=0.6, label="Passengers")
plt.scatter(new_pass[0,0], new_pass[0,2], c='gold', edgecolors='black', s=200, label="New Passenger")
plt.title("Passenger Age vs Pclass (color = survival)")
plt.xlabel("Age")
plt.ylabel("Pclass")
plt.legend()
