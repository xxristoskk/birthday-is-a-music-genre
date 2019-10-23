from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

### psudocode for possible recommendation system using the mcmc metropolis algorithm
""" user input: artists/genres
    there are two neighbors for each artist/genre
    randomly generate a proposal genre (do we go left or right)
    consider the 'population' of the proposal (p) genre and the 'population' of the input (i) genres
    if p > i, then move to p
    if p < i, then i - p
    place the remaining i with p in a 'bag' (maybe a placeholder list) and randomly chose one
    if the random pick is p, then go to p
    if the random pick is i, then stay at i
"""
# Takes in a dictionary of genres and a list of genres
# Finds the closest genre neighbors and adds them to a new list (tuned)
# The tuned list is checked for duplicates and a final list is returned
def genre_tuner(genre_dict, genres):
    genre_tuples = []
    tuned = []
    final = []
    for genre in genres:
        if genre not in dictionary.keys():
            print('Nothing found')
        else:
            genre_tuples.append(sorted(dictionary[genre].items(),key=lambda tup: tup[1],reverse=True)[:10]) # sorts tuples in decsending order
    for items in genre_tuples:
        i=50
        for tup in items:
            if tup[1] in range(0,i) and tup[0] not in genres:
                i = tup[1]
                tuned.append(tup[0])
    #Checks for duplicates
    for n in tuned:
        if n not in final:
            final.append(n)
    print(genre_tuples)
    return final

# Takes in list of releases as dictionaries, along with a list of genres
# Finds the neighboring genres and if both the neighbors and listed genres are in the releases genre, it is appended to new list
def curated_data(data, genres):
    genres = [i.lower() for i in genres]
    neighbors = genre_tuner(genre_dict_builder(data),genres)
    new = []
    for release in data:
        for neighbor in neighbors:
            for genre in genres:
                if neighbor not in release['genres']:
                    continue
                elif neighbor in release['genres'] and genre in release['genres']:
                    new.append(release)
    new = remove_duplicates(new)
    return new

e_df.shape
######### 166,192 electronic songs
####### KMEANS on electronic music ########
pickle.dump(e_df,open('electronic_dataframeFINAL.pickle','wb'))
e_df = pickle.load(open('electronic_dataframeFINAL.pickle','rb'))
e_df.columns
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

X2 = scaler.fit_transform(e_df.drop(columns=['genre','key','time_signature','liveness','analysis_url','duration_ms','id','mode','type','uri','track_href']))
kmeans = KMeans(n_clusters=7,n_init=500,max_iter=1000).fit(X2)
kmeans_pred = kmeans.fit_predict(X)
import seaborn as sns
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

km_df = pd.DataFrame(X2)
e_df.drop(columns=['genre','key','time_signature','liveness','analysis_url','duration_ms','id','mode','type','uri','track_href']).columns
km_df['labels'] = labels

##### adding labels to e_df #####
e_df['labels'] = labels
e_df_with_labels = e_df[['labels','id']]
e_df_with_labels
pickle.dump(e_df,open('labeled_e_df.pickle','wb'))
explore = e_df[['labels','genre','energy','danceability','loudness','acousticness']]
explore.drop_duplicates(inplace=True)
explore[explore['labels']==1][50:100]
explore.dropna(inplace=True)
explore['labels'].hist()
km_df.rename(columns={0:'acousticness',1:'danceability',2:'energy',3:'instrumentalness',
                      4:'loudness',5:'speechiness',6:'tempo',7:'valence'},inplace=True)
km_df.columns
######## make the heatmap bigger
sns.heatmap(km_df.groupby('labels').median(),xticklabels=True,annot=True)
plt.savefig('label_features.png')
""" Labeling clusters:
    0) Mellow bops
    1) Heavy bops
    2) Crying on the dancefloor
    3) Happy on the dancefloor
    4) Ahhhh I love words!
    5) Redbull & Vodka
    6) You're scaring all the bros
"""
km_df.shape
km_df = pd.DataFrame(X)
e_df.columns
km_df['labels'] = labels
km_df.rename(columns={0:'acousticness',1:'danceability',2:'energy',3:'instrumentalness',
                      4:'liveness',5:'loudness',6:'speechiness',7:'tempo',8:'valence'},inplace=True)
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

################## the recommendation engine ######################################
from surprise import Dataset, Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split
import functions as f
reader = Reader(rating_scale=(0,1))
train,test = train_test_split(df,test_size=.2)
