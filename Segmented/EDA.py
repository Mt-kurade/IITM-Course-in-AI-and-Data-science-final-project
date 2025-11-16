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