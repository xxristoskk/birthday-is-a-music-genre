import pandas as pd
import json
import functions as f
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

######################
#### NEW DATASET ####
bc_artist_data = pickle.load(open('new_bc_adata','rb'))

def get_bc_artist_info(loa):
    open('new_bc_cnfrm_pt2.pickle','ab')
    case_list = []
    bc_dict = {}
    count = 0
    for artist in tqdm(loa[266900:]):
        try:
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
        except:
            print('no longer on spotify')
            f.refresh_token()
        count +=1
        if count == 100:
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
    feat_data = []
    counter = 0
    for track in tqdm(trax):
        r = f. get_features(track)
        if len(r) > 1:
            feat_data.append(r)
            counter+=1
        else:
            continue
            counter+=1
        if counter == 50:
            f.refresh_token()    
    return feat_data


f.get_features(track_ids[0])
f.refresh_token()
###### connecting to mongodb #####
import config
import pymongo
from pymongo import MongoClient
client = MongoClient('mongodb://addy:config.mongo_pw@bc01-shard-00-00-muwwi.gcp.mongodb.net:27017,bc01-shard-00-01-muwwi.gcp.mongodb.net:27017,bc01-shard-00-02-muwwi.gcp.mongodb.net:27017/test?ssl=true&replicaSet=BC01-shard-0&authSource=admin&retryWrites=true&w=majority')
db = client['BC01']
collection = db['spotData.bc']
posts = db.posts
## inserting data ##
post_id = posts.insert_many(data)



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
