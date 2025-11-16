import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

matplotlib.use("Agg")
df = pd.read_csv("titanic_data.csv")

df['Gender_enc'] = df['Gender'].map({'Male':0, 'Female':1})
df['Embarked_enc'] = df['Embarked'].map({'C':0,'Q':1,'S':2})

avg_age = df['Age'].mean()
pclass_highest = df['Pclass'].value_counts().idxmax()
travel_alone_count = df['TravelingAlone'].sum()
survival_pct = df['Survived'].mean()*100
better_group = df.groupby('Gender')['Survived'].mean().idxmax()
embarked_S = (df['Embarked']=='S').sum()

under18 = df[df['Age']<18]
pclass_under18_best = under18.groupby('Pclass')['Survived'].mean().idxmax()
best_comb = df.groupby(['Gender','Pclass'])['Survived'].mean().idxmax()

print("\nSection 1: EDA ")
print("Average age:", avg_age)
print("Closest option:", min([25,30,35,40], key=lambda x:abs(x-avg_age)))
print("Pclass with highest passengers:", pclass_highest)
print("Passengers traveling alone:", travel_alone_count)
print("Survival percentage:", survival_pct)
print("Better survival group:", better_group)
print("Embarked from Southampton (S):", embarked_S)
print("Under 18 best survival Pclass:", pclass_under18_best)
print("Best Gender+Pclass combination:", best_comb)

plt.figure(figsize=(12,6))

# Age distribution
plt.subplot(2,2,1)
df['Age'].hist(bins=15, edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

# Survival by gender
plt.subplot(2,2,2)
df.groupby('Gender')['Survived'].mean().plot(kind='bar', color=['blue','pink'])
plt.title("Survival Rate by Gender")
plt.ylabel("Survival Rate")

# Pclass distribution
plt.subplot(2,2,3)
df['Pclass'].value_counts().sort_index().plot(kind='bar', color='green')
plt.title("Passenger Count by Pclass")
plt.xlabel("Pclass")

# Survival percentage pie
plt.subplot(2,2,4)
df['Survived'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=['Did Not Survive','Survived'])
plt.title("Survival Percentage")

plt.savefig("eda_plots.png")

plt.tight_layout()
#plt.show()


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

print("\n--- Section 2: K-NN ---")
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

print("\n--- Section 3: K-Means ---")
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
