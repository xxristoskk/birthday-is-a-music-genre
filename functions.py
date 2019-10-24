import streamlit as st
import spotipy
import json
import spotipy.util as util
from spotipy import oauth2
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import config
import time
from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

################### Spotify autho ######################
scope = 'playlist-modify-public'

def is_token_expired(token_info):
    now = int(time.time())
    return token_info['expires_at'] - now < 60

oauth = SpotifyOAuth(client_id=config.ClientID,client_secret=config.ClientSecret,redirect_uri='http://localhost/',scope=scope)
token_info = oauth.get_cached_token()
if not token_info:
    auth_url = oauth.get_authorize_url()
    print(auth_url)
    response = input('Paste the above link into your browser, then paste the redirect url here: ')
    code = oauth.parse_response_code(response)
    token_info = oauth.get_access_token(code)
    token = token_info['access_token']

sp = spotipy.Spotify(auth=token)


####### function to flatten out lists of lists ######
def flatten_lists(list_of_lists):
    return [x for y in list_of_lists for x in y]

################## functions ###############
def is_token_expired(token_info):
    now = int(time.time())
    return token_info['expires_at'] - now < 60

def refresh_token():
    global token_info, sp
    if is_token_expired(token_info):
        token_info = oauth.refresh_access_token(token_info['refresh_token'])
        token = token_info['access_token']
        sp = spotipy.Spotify(auth=token)

def find_song(artist,song):
    try:
        id_ = sp.search(q=f'artist:{artist} track:{song}',type='track')['tracks']['items'][0]['id']
        return id_
    except:
        st.write("Couldn't find what you're looking for.")

def get_features(id_):
    return sp.audio_features(id_)

def normalize_col(df):
    for col in df:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def model_work(artist,song,model):
    scaler = StandardScaler()
    id_ = find_song(artist,song)
    try:
        df = pd.DataFrame(get_features(id_))
        df.drop(columns=['key','time_signature','liveness','analysis_url','duration_ms','id','mode','type','uri','track_href'],inplace=True)
        X = scaler.fit_transform(df)
        return model.predict(X)
    except:
        st.write("This song doesn't have audio features available :(")


def search_db(data,class_):
    id_list = []
    inds = np.random.randint(low=0, high=10000, size=15)
    for i in range(15):
        id_list.append(data[data['labels']==class_]['id'][i])
    return id_list

def write_the_results(id_list):
    r = sp.tracks(id_list)
    list_of_anames=[x['album'][0]['name'] for x in r['tracks']]
    list_of_tnames=[x['name'] for x in r['tracks']]
    loa = flatten_lists(list_of_anames)
    lot = flatten_lists(list_of_tnames)
    zipped = list(tup(loa,lot))
    for item in zipped:
        st.write(f'"{item[1]}" by {item[0]}')
