import pandas as pd
import json
import functions as f
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
    open('songs7.pickle','ab')
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
            pickle.dump(feat_data,open('songs7.pickle','wb'))
            feat_data = pickle.load(open('songs7.pickle','rb'))
            time.sleep(.25)
            dump = 0
    return

###### getting the first 250 ######
get_new_feats(track_ids[1000000:1500000])
944776+86300
l = pickle.load(open('songs7.pickle','rb'))
len(l)
###### getting the next 250 #####
s1 = pickle.load(open('songs.pickle','rb'))
s2 = pickle.load(open('songs2.pickle','rb'))
s3 = pickle.load(open('songs3.pickle','rb'))
s4 = pickle.load(open('songs4.pickle','rb'))
s5= pickle.load(open('songs5.pickle','rb'))
s6= pickle.load(open('songs6.pickle','rb'))
s7= pickle.load(open('songs7.pickle','rb'))

songs = s1+s2+s3+s4+s5+s6+s7
songs = f.flatten_lists(songs)

for song in songs:
    if song == None:
        songs.remove(song)

af_df = pd.DataFrame(songs)

af_df.info()

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

#### newly scraped bc artist info based on genre
diw2 = pickle.load(open('bc_genre_releases2.pickle','rb'))
diw3 = pickle.load(open('bc_genre_releases3.pickle','rb'))
diw1 = pickle.load(open('bc_genre_releases.pickle','rb'))
all_releases = diw3 + diw2 + diw1
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


####### scaling ##########
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(e_df)
e_df.isna().sum()

###Perform PCA and plot results
from sklearn.decomposition import PCA
pca = PCA(n_components=11)
pca_data = pca.fit_transform(X)
df_pca = pd.DataFrame(data=pca_data,columns=[1,2,3,4,5,6,7,8,9,10,11])

import seaborn as sns
plt.scatter(pca_data[:,0],pca_data[:,2])
plt.plot(np.cumsum(pca.explained_variance_ratio_))

sns.heatmap(e_df.corr())
index = np.arange(11)
plt.bar(index, pca.explained_variance_ratio_)
plt.title('Scree plot for PCA')
plt.xlabel('Num of components')
plt.ylabel('proportion of explained variance')
df_pca
fig = plt.figure(figsize=(13,11))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_pca[1], df_pca[2], df_pca[3], c='green')
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
