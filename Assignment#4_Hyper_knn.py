import seaborn as sns
import pandas as pd

# sns.get_dataset_names()
exercise = sns.load_dataset("exercise")
df = pd.DataFrame(exercise)
df = df.drop("Unnamed: 0", axis=1)
df['time'] = df['time'].str.replace(' min', '') # 원래 time에 min이있어서 다잘라줌
df['diet'] = df['diet'].str.replace('low fat', '0')
df['diet'] = df['diet'].str.replace('no fat', '1')
df['kind'] = df['kind'].str.replace('rest', '0')
df['kind'] = df['kind'].str.replace('walking', '1')
df['kind'] = df['kind'].str.replace('running', '2')
t = df["kind"]
df = df.drop("kind", axis=1)

from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(df,t,test_size = 0.3 , random_state=42,stratify =t)

score = []
score2 = []
best_score = 0

from sklearn.neighbors import KNeighborsClassifier
for n in [1,2,3,4,5,6,7,8,9,10]:
  for m in [1]:
    kn = KNeighborsClassifier(n_neighbors = n , p =m  )
    kn.fit(train_data,train_target)
    val_score = kn.score(test_data,test_target)
    score.append(val_score)
print(score)
for n in [1,2,3,4,5,6,7,8,9,10]:
  for m in [2]:
    kn = KNeighborsClassifier(n_neighbors = n , p =m  )
    kn.fit(train_data,train_target)
    val_score = kn.score(test_data,test_target)
    score2.append(val_score)
print(score2)
import matplotlib.pyplot as plt
n_values = [1,2,3,4,5,6,7,8,9,10]
plt.plot(n_values,score)
plt.plot(n_values,score2)
plt.xlabel('n_neighbors')
plt.ylabel('score')
plt.show()
