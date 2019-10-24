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

#### bc_data ###
bcdf = pickle.load(open('/home/xristsos/Documents/nodata/curation_station/bc_feat_df.pickle','rb'))
bcdf
######### 166,192 electronic songs
import pickle
####### KMEANS on electronic music ########
# pickle.dump(e_df,open('electronic_dataframeFINAL.pickle','wb'))
e_df = pickle.load(open('electronic_dataframeFINAL.pickle','rb'))
e_df = pd.concat([bcdf,e_df])

e_df.shape
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

import seaborn as sns
labels = kmeans.labels_

km_df = pd.DataFrame(X)
e_df.drop(columns=['genre','key','time_signature','liveness','analysis_url','duration_ms','id','mode','type','uri','track_href']).columns
km_df['labels'] = labels
e_df.shape
e_df.drop_duplicates(inplace=True)
e_df.dropna(inplace=True)
e_df[e_df['labels']==6][['labels','genre']][:60]

##### adding labels to e_df #####
e_df['labels'] = labels
pickle.dump(e_df,open('everything_db.pickle','wb'))
e_df_with_labels = e_df[['labels','id']]

# explore = e_df[['labels','genre','energy','danceability','loudness','acousticness']]
# explore.drop_duplicates(inplace=True)
# explore[explore['labels']==1][50:100]
# explore.dropna(inplace=True)
# explore['labels'].hist()
km_df.rename(columns={0:'acousticness',1:'danceability',2:'energy',3:'instrumentalness',
                      4:'loudness',5:'speechiness',6:'tempo',7:'valence'},inplace=True)
km_df.columns
######## make the heatmap bigger
plt.tight_layout(.15)
sns.heatmap(km_df.groupby('labels').mean(),xticklabels=True,annot=True)

plt.tight_layout(.15)
sns.heatmap(km_df.groupby('labels').median(),xticklabels=True,annot=True)
plt.savefig('label_features.png')
""" Labeling clusters:
    0) Its art but it hurts
    1) Everything Henry Rollins Hates*
    2) Why am I crying in the club right now
    3) Soft vibes
    4) Redbull & Vodka
    5) Big club energy
    6) Spacy bassy
"""

# centers, labels = find_clusters(X, 6, 10)
fig = plt.figure(figsize=(13,11))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,2], X[:, 6], X[:,8], c=labels)
plt.savefig('energy-insturm-valence2.png')
plt.show()

############## classifing the clusters ##########################
target = km_df['labels']
features = km_df.drop(columns='labels')
xTrain,xTest,yTrain,yTest = train_test_split(features,target,test_size=.3,random_state=5)
rfc = RandomForestClassifier(criterion='gini',n_estimators=120,max_depth=5,min_samples_leaf = 0.05)
rfc.fit(xTrain,yTrain)
rfc_pred = rfc.predict(xTest)

###### scores ######
recall_score(yTest,rfc_pred,average='weighted')
f1_score(yTest,rfc_pred,average='weighted')
precision_score(yTest,rfc_pred,average='weighted')
accuracy_score(yTest,rfc_pred)
confusion_matrix(yTest,rfc_pred)
