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
e_df = pickle.load(open('electronic_dataframeFINAL.pickle','rb'))
e_df = pd.concat([bcdf,e_df])

# Specifying the dataset and initializing variables
distorsions = []

# Calculate SSE for different K
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state = 10)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)

# Plot values of SSE
plt.figure(figsize=(15,8))
plt.subplot(121, title='Elbow curve')
plt.xlabel('k')
plt.plot(range(1, 10), distorsions)
plt.grid(True)
plt.savefig('elbow curve.png')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(e_df.drop(columns=['genre','key','time_signature','liveness','analysis_url','duration_ms','id','mode','type','uri','track_href']))
kmeans = KMeans(n_clusters=7,n_init=500,max_iter=1000).fit(X)
kmeans_pred = kmeans.fit_predict(X)
pickle.dump(kmeans,open('kmeans_fit.pickle','wb'))

labels = kmeans.labels_

km_df = pd.DataFrame(X)
e_df.drop(columns=['genre','key','time_signature','liveness','analysis_url','duration_ms','id','mode','type','uri','track_href']).columns
km_df['labels'] = labels
e_df.drop_duplicates(inplace=True)
e_df.dropna(inplace=True)
e_df[e_df['labels']==6][['labels','genre']][:60]
e_df.drop(columns='genre',inplace=True)

##### adding labels to e_df #####
e_df['labels'] = labels
# pickle.dump(e_df,open('everything_db.pickle','wb'))
e_df_with_labels = e_df[['labels','id']]
e = pickle.load(open('everything_db.pickle','rb'))
e.reset_index(drop=True,inplace=True)

fig = px.histogram(e, x="labels", nbins=20)
fig.show()

import plotly.express as px
fig = px.histogram(e, x="labels", color="labels", marginal="rug")
fig.show()



km_df.rename(columns={0:'acousticness',1:'danceability',2:'energy',3:'instrumentalness',
                      4:'loudness',5:'speechiness',6:'tempo',7:'valence'},inplace=True)


#### Heatmap that describes the clusters' most prominent features

plt.tight_layout(1)
sns.heatmap(km_df.groupby('labels').median(),xticklabels=True,annot=True)
plt.savefig('label_features.png')

## 3D scatterplot of the clusters
fig = plt.figure(figsize=(13,11))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,2], X[:, 3], X[:,7], c=labels)
plt.savefig('energy-insturm-valence3.png')
plt.show()

km_df.drop_duplicates(inplace=True)

############## classifing the clusters with random forest ##########################
target = km_df['labels']
features = km_df.drop(columns='labels')
xTrain,xTest,yTrain,yTest = train_test_split(features,target,test_size=.3)
# rfc = RandomForestClassifier(n_estimators=1000,max_features=3,max_depth=11)
trained_rfc = RandomForestClassifier(n_estimators=1000,max_features=3,max_depth=11).fit(features,target)
pickle.dump(trained_rfc,open('trained_rfc.pickle','wb'))
rfc_pred = rfc.predict(xTest)

###### scores ######
recall_score(yTest,rfc_pred,average='weighted')
f1_score(yTest,rfc_pred,average='weighted')
precision_score(yTest,rfc_pred,average='weighted')
accuracy_score(yTest,rfc_pred)
confusion_matrix(yTest,rfc_pred)

from sklearn.model_selection import GridSearchCV
def grid_search(xTrain,xTest,yTrain,yTest):
    gs = GridSearchCV(estimator=RandomForestClassifier(),
                     param_grid={'max_depth': [3,8,11],
                                 'n_estimators': (25,50,75,100,500,1000),
                                 'max_features': (1,3,5,7)},
                     cv=4,n_jobs=-1,scoring='balanced_accuracy')
    model = gs.fit(xTrain,yTrain)
    print(f'Best score: {model.best_score_}')
    print(f'Best parms: {model.best_params_}')

grid_search(xTrain,xTest,yTrain,yTest)

"""
GridSearch results
Best score: 0.9292720520606348
Best parms: {'max_depth': 11, 'max_features': 3, 'n_estimators': 1000}
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

plot_feature_importances(rfc,xTrain)
