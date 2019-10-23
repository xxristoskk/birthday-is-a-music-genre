import pandas as pd
import json
# import functions as f
import pickle
import matplotlib.pyplot as plt
import numpy as np

df = pickle.load(open('this thing.pickle','rb'))
df['genre'] = 'electronic'
df.shape
df1 = pickle.load(open('FINAL_G_FEAT_DF.pickle','rb'))
df1.shape
df.drop(columns=['analysis_url','duration_ms','id','mode','track_href','type','uri'],inplace=True)
df1.drop(columns=['analysis_url','duration_ms','id','mode','track_href','type','uri'],inplace=True)
df = pd.concat([df,df1])
df.shape
# df.isna().sum()
# df['genre'].nunique()
# #### 1072 unique genres
# log = df['genre'].unique()
# list(log)
#
# df.drop_duplicates(inplace=True)
# df_no_g = df.drop(columns='genre')
# sns.heatmap(e_df.corr())
# ##### dropping key since it is not a feature that distinguishes any difference between styles of music
# # df.drop(columns='key',inplace=True)
# ##### creating different dataframes for specific genres give a more in tune visualization
# rock_df = pd.concat([df[df['genre'] == 'metal'],df[df['genre']=='punk'],df[df['genre']=='rock'],df[df['genre']=='blues'],df[df['genre']=='folk'],
#                      df[df['genre']=='surf'],df[df['genre']=='alternative'],df[df['genre']=='grunge'],df[df['genre']=='post-post-hardcore'],
#                      df[df['genre']=='hardcore']])
#
# rock_df.drop(columns='genre',inplace=True)
# rock_df.shape
# rock_df.reset_index(inplace=True,drop=True)
# df.head()
#
# with pd.option_context('display.max_rows', 1000, 'display.max_columns', 5):  # more options can be specified also
#     print(df)

######## electronic dataframe ################
e_df = pd.concat([
                           df[df['genre']=='electro'],
                           df[df['genre']=='idm'],
                           df[df['genre']=='glitch'],
                           df[df['genre']=='grime'],
                           df[df['genre']=='dub techno'],
                           df[df['genre']=='trap'],
                           df[df['genre']=='remix'],
                           df[df['genre']=='lo-fi beats'],
                           df[df['genre']=='chillhop'],
                           df[df['genre']=='meme rap'],
                           df[df['genre']=='underground hip hop'],
                           df[df['genre']=='bass music'],
                           df[df['genre']=='moog'],
                           df[df['genre']=='substep'],
                           df[df['genre']=='afro house'],
                           df[df['genre']=='vapor house'],
                           df[df['genre']=='jersey club'],
                           df[df['genre']=='rave'],
                           df[df['genre']=='hard house'],
                           df[df['genre']=='lo-fi house'],
                           df[df['genre']=='electra'],
                           df[df['genre']=='minimal dub'],
                           df[df['genre']=='experimental techno'],
                           df[df['genre']=='deep techno'],
                           df[df['genre']=='electronic'],
                           df[df['genre']=='techno'],
                           df[df['genre']=='acid'],
                           df[df['genre']=='house'],
                           df[df['genre']=='footwork'],
                           df[df['genre']=='juke'],
                           df[df['genre']=='rap'],
                           df[df['genre']=='vaporwave'],
                           df[df['genre']=='trance'],
                           df[df['genre']=='hip hop'],
                           df[df['genre']=='deep house'],
                           df[df['genre']=='ambient'],
                           df[df['genre']=='club'],
                           df[df['genre']=='breakbeat'],
                           df[df['genre']=='bass'],
                           df[df['genre']=='synth pop'],
                           df[df['genre']=='desi hip hop'],
                           df[df['genre']=='tamil hip hop'],
                           df[df['genre']=='drill'],
                           df[df['genre']=='electropop'],
                           df[df['genre']=='hopebeat'],
                           df[df['genre']=='electronica'],
                           df[df['genre']=='electro dub'],
                           df[df['genre']=='wonky'],
                           df[df['genre']=='acid idm'],
                           df[df['genre']=='west coast rap'],
                           df[df['genre']=='bass house'],
                           df[df['genre']=='bass trap'],
                           df[df['genre']=='brostep'],
                           df[df['genre']=='chillstep'],
                           df[df['genre']=='tech house'],
                           df[df['genre']=='acid techno']

])
# df.reset_index(inplace=True,drop=True)
e_df.isna().sum()
e_df.shape
# e_df.drop(columns=['key','genre'],inplace=True)
e_df.drop(columns='genre',inplace=True)
############################################
###########second dataframe ####################
#################################################
e_df2 = pickle.load(open('FINAL LIST OF FEATURES FOR ELECTRONIC MUSIC.pickle','rb'))
len(e_df2)
for x in e_df2:
    if x == None:
        e_df2.remove(x)
    # if x.keys():
    #     continue
    # else:
    #     e_df2.remove(x)
len(e_df2)
e2 = pd.DataFrame(e_df2)
e2.shape
e2.isna().sum()
e2.drop(columns=['analysis_url','duration_ms','id','mode','type','uri','track_href'],inplace=True)
e_df = pd.concat([e2,e_df])

e_df.shape

####### scaling all the tempos to be between 60-120 by double or halfing the tempos ######
# def tempo_classifier(df):
#     for i,row in df.iterrows():
#         if row['tempo'] >= 120:
#             row['tempo'] = row['tempo'] / 2
#         elif row['tempo'] <= 60:
#             row['tempo'] = row['tempo'] * 2
#         else:
#             continue
#     return df

####### scaling ##########
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(e_df)
e_df.isna().sum()
###Calculate PCA and plot results
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

x##### one-hot encoding #####
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
labeld_genres = encoder.fit_transform(main_df['genre'])
main_df['encoded_genres'] = labeld_genres
encoded_df = main_df.drop(columns=['genre'])

##### some quick plotting
import plotly.graph_objects as go

fig = go.Figure(data=go.Heatmap(
                   z=df.corr(),
                   x=list(df.columns),
                   y=list(df.columns)))
fig.show()
##### dropping liveness due to its high correlation with danceability
df.drop(columns='liveness',inplace=True)

##### interesting stuff i can't look at right now
fig2 = go.Figure(data=go.Heatmap(
                   z=df.groupby('genre').mean().corr(),
                   x=list(df.columns),
                   y=list(df.columns)))
fig2.show()

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
