import pandas as pd
# import json
# import functions as f
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

######################
#### NEW DATASET ####
bc_artist_data = pickle.load(open('new_bc_adata','rb'))

def get_bc_artist_info(loa):
    open('new_bc_cnfrm_pt2.pickle','ab')
    case_list = []
    bc_dict = {}
    count = 0
    for artist in tqdm(loa[266900:]):
        r = f.find_artist(f'"{artist}"')['artists']
        if len(r['items']) < 1:
            continue
        else:
            r = r['items'][0]
        bc_dict = {'followers': r['followers']['total'],
                   'genres': r['genres'],
                   'id': r['id'],
                   'artist_name': r['name'],
                   'popularity': r['popularity'],
                   'top_trax': f.get_top_tracks(r['id'])}
        case_list.append(bc_dict)
        f.refresh_token()
    if count == 25:
            pickle.dump(case_list,open('new_bc_cnfrm_pt2.pickle','wb'))
            case_list = pickle.load(open('new_bc_cnfrm_pt2.pickle','rb'))
            count = 0
    return

get_bc_artist_info(bc_artist_data)

############# 500k+ bandcamp artist info ###########
data = pickle.load(open('bc_dicts_from_spotify.pickle','rb'))
df = pd.DataFrame(data)
data[0]

###### getting all the track ids for audio features ####
track_ids = []
for item in data:
    if len(item['top_trax']) < 1:
        continue
    else:
        track_ids.append(item['top_trax'])
track_ids = f.flatten_lists(track_ids)

def get_new_feats(trax):
    open('big_batch.pickle','ab')
    feat_data = []
    dump = 0
    for track in tqdm(trax):
        try:
            r = f. get_features(track)
        except:
            continue
        if r != 'None':
            feat_data.append(r)
            dump+=1
        else:
            dump+=1
        f.refresh_token()
        if dump == 25:
            pickle.dump(feat_data,open('big_batch.pickle','wb'))
            feat_data = pickle.load(open('big_batch.pickle','rb'))
            time.sleep(.25)
            dump = 0
    return
get_new_feats(track_ids[2266101:])

a = pickle.load(open('song_batch1.pickle','rb'))
l1 = []
l2 = []
for song in songs:
    if song['id'] not in l1:
        l1.append(song)
    else:
        l2.append(song)

for song in songs:
    if song == None:
        songs.remove(song)

###### connecting to mongodb #####
import config
import pymongo
from pymongo import MongoClient
import dns

client = MongoClient(f'mongodb+srv://xristos:{config.mongo_pw}@bc01-muwwi.gcp.mongodb.net/test?retryWrites=true&w=majority')
db = client.BC01

## creating a new collection
audioFeatures = db['audioFeatures']
artistInfo = db['artistInfo']
## inserting data ##
results_a = audioFeatures.insert_many(songs)
results_i = artistInfo.insert_many(data)
db.list_collection_names()


#### clean and organize scraped data
l = []
for item in all_releases:
    for g,v in item.items():
        z = list(zip(list(v.keys()),list(v.values())))
        for x in z:
            nd = {'genre':g,
                  'artist':x[0],
                  'album': x[1]}
            l.append(nd)
#### finding all bc artists with no genre info
artists_with_no_genres = artistInfo.find({'genres':[]},{'artist_name':1})

#### creating a reference list to find out which artists need to be updated in the database
artist_query = []
for artist in artists_with_no_genres:
    artist_query.append(artist['artist_name'])

#### updating each artist with new genre info
for item in tqdm(l):
    if item['artist'] in artist_query:
        artistInfo.update_one({'artist_name': item['artist']},{'$set':{'genres':item['genre']}})

""" the above needs more scraping to be done; there are over 400k artists without genre info (from spotify) and from
the genre pages i scraped on bandcamp, only around 30k were found. the next step will be to write another scraping script
which will search for the specific artists on bc and grab the genre tags directly from the artist page """

####### audio features dataframe ########
df = pickle.load(open('feat_df.pickle','rb'))
df.drop_duplicates(inplace=True)
df.shape
df.columns
df.drop(columns=['duration_ms','id','key','mode','time_signature','uri','track_href','type','analysis_url'],inplace=True)
df.head(10)

techno_df = df[df['genre']=='techno']
techno_df = techno_df[techno_df['acousticness']<.5]
techno_df = techno_df[techno_df['liveness']<.5]
techno_df.shape
techno_df.shape

plt.figure(figsize=(15,10))
df['genre'].value_counts().plot(kind='bar')
plt.savefig('bc_genre_dist.png')

import seaborn as sb
sb.heatmap(techno_df.corr())
df.shape
####### scaling ##########
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df.drop(columns='genre'))

###Perform PCA and plot results
from sklearn.decomposition import PCA
pca = PCA(n_components=9)
pca_data = pca.fit_transform(X)
df_pca = pd.DataFrame(data=pca_data,columns=[1,2,3,4,5,6,7,8,9])

plt.scatter(pca_data[:,0],pca_data[:,1])
plt.plot(np.cumsum(pca.explained_variance_ratio_))

sns.heatmap(e_df.corr())
index = np.arange(9)
plt.bar(index, pca.explained_variance_ratio_)
plt.title('Scree plot for PCA')
plt.xlabel('Num of components')
plt.ylabel('proportion of explained variance')
df_pca
fig = plt.figure(figsize=(13,11))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_pca[1], df_pca[5], df_pca[3], c='green')
plt.show()

##### some quick plotting
import plotly.graph_objects as go

fig = go.Figure(data=go.Heatmap(
                   z=df.corr(),
                   x=list(df.columns),
                   y=list(df.columns)))
fig.show()
##### dropping liveness due to its high correlation with danceability
df.drop(columns='liveness',inplace=True)


# ###### genre matrix eda ######
# g_dict = json.load(open('/home/xristsos/flatiron/projects/offsite final/test_genre_dict2.json','r'))
# g_df = pd.DataFrame(g_dict)
# g_df[g_df['techno'] == 2]['techno'] ### look at this -- there is acoustic and alternative rock with techno
# ###### tech house and grime are also in that same group. there will need to be audio features to filter out this outliers
#
# g_df['rap'].sum()
# g_df['hip hop'].sum()
# g_df['pop'].sum()
#
# ##### calculating the percentages #####
# a_dict = {}
# for genre in g_df.columns:
#     a_dict[genre] = g_df[genre]/g_df[genre].sum()
# a_dict
# genre_prob_df = pd.DataFrame(a_dict)
# ####### using a scaler instead #########
# ###min-max function####
# def minmax(row):
#     new = (row - row.mean())/(row.max()-row.min())
#     return new
# # genre_prob_df.describe()
# std_genre.fillna(0,inplace=True)
# gp = g_df.T
# gp.reset_index(inplace=True)
# gp['index'] = gp['index'].sort_values()
# gsort = gp['index'].sort_values()
# gp.drop(columns='index',inplace=True)
# gp.set_index(gsort,inplace=True)
# gp.head()
