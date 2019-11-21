from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import pickle

#### bc_data ###
bcdf = pickle.load(open('/home/xristsos/Documents/nodata/curation_station/bc_feat_df.pickle','rb'))
######### 166,192 electronic songs
####### KMEANS on electronic music ########
bcdf.acousticness.sort_values()
e_df = pickle.load(open('/home/xristsos/flatiron/projects/birthday pickles/electronic_dataframeFINAL.pickle','rb'))
sns.heatmap(e_df.corr())



# Specifying the dataset and initializing variables
distorsions = []
# Calculate SSE for different K
for k in tqdm(range(1, 10)):
    kmeans = KMeans(n_clusters=k, random_state = 10)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)

# Plot values of SSE
plt.figure(figsize=(15,8))
plt.subplot(121, title='Elbow curve')
plt.xlabel('k')
plt.plot(range(1, 10), distorsions)
plt.grid(True)
plt.savefig('elbow curve(techno).png')

genres = pickle.load(open('bandcamp_genres.pickle','rb'))

def cluster_all_genres(df,genres):
    scaler = StandardScaler()
    for genre in tqdm(genres):
        ndf = df[df['genre']==genre]
        X = scaler.fit_transform(ndf.drop(columns='genre'))
        kmeans = KMeans(n_clusters=7,n_init=50,max_iter=100).fit(X)
        pickle.dump(kmeans,open(f'kmeans_{genre}.pickle','wb'))
    return

cluster_all_genres(df,genres)

def classify_all_genres(df,genre):
    
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df[df['genre']=='techno'].drop(columns='genre'))

kmeans = KMeans(n_clusters=8,n_init=50,max_iter=100).fit(X)
kmeans_pred = kmeans.fit_predict(X)
pickle.dump(kmeans,open('kmeans_fit_all.pickle','wb'))

labels = kmeans.labels_

km_df = pd.DataFrame(X)
km_df['labels'] = labels


##### adding labels to e_df #####
df['labels'] = labels
df['labels'].value_counts().plot(kind='bar')


km_df.rename(columns={0:'acousticness',1:'danceability',2:'energy',3:'instrumentalness',4:'liveness',
                      5:'loudness',6:'speechiness',7:'tempo',8:'valence'},inplace=True)

km_df.columns
#### Heatmap that describes the clusters' most prominent features
e_df.columns
plt.tight_layout(1)
sns.heatmap(km_df.groupby('labels').median(),xticklabels=True,annot=True)
plt.savefig('label_features_everything.png')

techno_df['labels'].value_counts().plot(kind='bar')

## 3D scatterplot of the clusters
fig = plt.figure(figsize=(13,11))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['energy'], df['danceability'], df['valence'], c=labels)
plt.savefig('energy-insturm-valence-new.png')
plt.show()

km_df.drop_duplicates(inplace=True)
import numpy as np
############## classifing the clusters with random forest ###############
target = df['labels']
features = df.drop(columns=['genre','labels'])
xTrain,xTest,yTrain,yTest = train_test_split(features,target,test_size=.3)
xTrain = scaler.fit_transform(xTrain)
xTest =  scaler.fit_transform(xTest)

# rfc = RandomForestClassifier(n_estimators=1000,max_features=3,max_depth=11)
trained_rfc = RandomForestClassifier(n_estimators=100,max_features=1,max_depth=8).fit(features,target)
pickle.dump(trained_rfc,open('trained_rfc_everything.pickle','wb'))
rfc_pred = trained_rfc.predict(xTest)

###### scores ######
recall_score(yTest,rfc_pred,average='weighted')
f1_score(yTest,rfc_pred,average='weighted')
precision_score(yTest,rfc_pred,average='weighted')
accuracy_score(yTest,rfc_pred)
confusion_matrix(yTest,rfc_pred)

from sklearn.model_selection import GridSearchCV
def grid_search(xTrain,xTest,yTrain,yTest):
    gs = GridSearchCV(estimator=RandomForestClassifier(),
                     param_grid={'max_depth': [3,5,8],
                                 'n_estimators': (25,50,75,100),
                                 'max_features': (1,3,5,8)},
                     cv=4,n_jobs=-1,scoring='balanced_accuracy')
    model = gs.fit(xTrain,yTrain)
    print(f'Best score: {model.best_score_}')
    print(f'Best parms: {model.best_params_}')

grid_search(xTrain,xTest,yTrain,yTest)

"""
GridSearch results
Best score: 0.9292720520606348
Best parms: {'max_depth': 8, 'max_features': 1, 'n_estimators': 75}
"""
import numpy as np
def plot_feature_importances(model,X_train):
    n_features = X_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train.columns.values)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.savefig('important_features.png')

plot_feature_importances(trained_rfc,xTrain)
