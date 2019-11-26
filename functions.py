import streamlit as st
import spotipy
import spotipy.util as util
from spotipy import oauth2
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import os
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pymongo
import config

##### Prepare the database
mongo_pw = os.environ['mongo_pw']
# mongo_pw = config.mongo_pw
client = pymongo.MongoClient(f'mongodb+srv://xristos:{mongo_pw}@bc01-muwwi.gcp.mongodb.net/test?retryWrites=true&w=majority')
db = client.BC01
artistInfo = db['artistInfo']


################### Spotify autho ######################
def is_token_expired(token_info):
    now = int(time.time())
    return token_info['expires_at'] - now < 60

def refresh_token():
    global token_info, sp
    if is_token_expired(token_info):
        token_info = oauth.refresh_access_token(token_info['refresh_token'])
        token = token_info['access_token']
        sp = spotipy.Spotify(auth=token)

scope = 'playlist-modify-public'
oauth = SpotifyOAuth(client_id=os.environ['ClientID'],client_secret=os.environ['ClientSecret'],redirect_uri='http://localhost/',scope=scope)
token_info = oauth.get_cached_token()
if not token_info:
    auth_url = oauth.get_authorize_url()
    st.write(auth_url)
    response = st.text_input('Click the above link, then paste the redirect url here and hit enter: ')
    st.write('*The BadAuth error below should go away after entering the redirect url')
    # response = input('Paste the above link into your browser, then paste the redirect url here: ')
    if response == "":
        time.sleep(5)
    code = oauth.parse_response_code(response)
    token_info = oauth.get_access_token(code)
    token = token_info['access_token']
    sp = spotipy.Spotify(auth=token)


## helper functions spotify api
def get_top_tracks(artist_id):
    return [x['id'] for x in sp.artist_top_tracks(artist_id)['tracks']]

def check_playlist(user, pl_name):
    for playlist in sp.user_playlists(user)['items']:
        if pl_name == playlist['name']:
            return playlist['id']
        else:
            return create_playlist(user,pl_name)['id']

def create_playlist(user,name):
    return sp.user_playlist_create(user,name)
def search_album(album):
    return sp.search(q='album:' + album, type='album')
def add_to_playlist(user, playlist_id, track_id):
    return sp.user_playlist_add_tracks(user, playlist_id, track_id)
def get_track_ids(album_id):
    return sp.album_tracks(album_id)['items'][0]['id']
def find_artist(artist_name):
    return sp.search(q='artist:' + artist_name, type='artist')

## Takes in a dictionary,username, and playlist name
## Returns a playlist with the first track from each album
def pl_creator(track_ids, user, pl_name):
    pl_id = check_playlist(user,pl_name)
    ## Search for albums in the dictionary
    add_to_playlist(user,pl_id,track_ids)
    return st.title('The playlist is done! (ﾉ☉ヮ⚆)ﾉ ⌒*:･ﾟ✧')

####### function to flatten out lists of lists ######
def flatten_lists(list_of_lists):
    return [x for y in list_of_lists for x in y]

########### app functions ###############

### finds the genre of the artist of the user's song
def find_genre(artist,song):
    global genre
    ##### define genres #####
    q = artistInfo.find({'bandcamp_genres':{'$exists':True}})
    genres = [x['bandcamp_genres'] for x in q]
    genres = flatten_lists(genres)
    #### find the spotify genres based on user's search ####
    artist_id = sp.search(q=f'artist:{artist} track:{song}',type='track')['tracks']['items'][0]['artists'][0]['id']
    spotify_genres = sp.artist(artist_id)['genres']
    possible_genre_matches = []
    for g in spotify_genres:
        if g in genres:
            possible_genre_matches.append(g)
    if len(possible_genre_matches) < 1:
        for gnr1 in genres:
            for gnr2 in spotify_genres:
                if gnr1 in gnr2:
                    possible_genre_matches.append(gnr1)
                    break
    if len(possible_genre_matches) < 1:
        return print('Nothing in the database for this yet (╥﹏╥)')
    genre = possible_genre_matches[0]
    return genre

### searchs for the user's song
def find_song(artist,song):
    try:
        id_ = sp.search(q=f'artist:{artist} track:{song}',type='track')['tracks']['items'][0]['id']
        return id_
    except:
        st.write("Couldn't find what you're looking for (╥﹏╥)")

### gets the audio features the user's song
def get_features(id_):
    return sp.audio_features(id_)

### normalizes the data for the data
def normalize_col(df):
    for col in df:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

### finds the user's song and runs its audio features through the trained random forest model ###
def model_work(artist,song,model):
    scaler = StandardScaler()
    id_ = find_song(artist,song)
    try:
        df = pd.DataFrame(get_features(id_))
        df.drop(columns=['key','time_signature','analysis_url','duration_ms','id','mode','type','uri','track_href'],inplace=True)
        X = scaler.fit_transform(df)
        return model.predict(X)
    except:
        st.write("This song doesn't have audio features available (╥﹏╥)")

### searches the database for songs by artists that are in both the same genre and class as the user's song ###
def search_db(class_,genre):
    class_ = class_.astype(str)
    class_ = flatten_lists(class_)
    q = artistInfo.find({'genres':genre,'class':class_[0]},{'top_trax':1})
    results = [x['top_trax'] for x in q]
    results = flatten_lists(results)
    if len(results) < 1:
        return st.write("Couldn't find what you're looking for (╥﹏╥)")
    results_length = len(results) - 1
    random_indicies = np.random.random_integers(0,results_length,40)
    id_list = []
    for i in random_indicies:
        id_list.append(results[i])
    return id_list

### class names ###
one = 'Redbull & Vodka'
two = 'Nap time'
three = "You're scaring the bros"
four = 'Rave church'
five = 'Sounds like a heart attack'
six = 'Why am I crying in the club rn'
seven = 'Just let me chill, damn'

#### takes in the list of track ids for the playlist and displays the results to the user ###
def display_results(track_ids,genre,p_class):
    ### display the category (class) of the user's song ###
    if p_class == 0:
        st.write(f'This song is in the "{one}" category')
    elif p_class == 1:
        st.write(f'This song is in the "{two}" category ')
    elif p_class == 2:
        st.write(f'This song is in the "{three}" category')
    elif p_class == 3:
        st.write(f'This song is in the "{four}" category')
    elif p_class == 4:
        st.write(f'This song is in the "{five}" category')
    elif p_class == 5:
        st.write(f'This song is in the "{six}" category')
    elif p_class == 6:
        st.write(f'This song is in the "{seven}" category')
    ### display the info for the songs in the playlist ###
    r = sp.tracks(track_ids)
    pop_artist = ""
    followers = 0
    song = ""
    for item,value in r.items():
        artist_id = value[0]['artists'][0]['id']
        artist_r = sp.artist(artist_id)
        name = artist_r['name']
        genres = artist_r['genres']
        link = artist_r['external_urls']['spotify']
        st.write(f'Artist name: {name}, Spotify genres: {genres}, Spotify link: {link}')
        f = artist_r['followers']['total']
        if f > followers:
            followers = f
            pop_artist = artist_r['name']
            song = value[0]['name']
        else:
            continue
    st.write(f'This search is looking for {genre} songs')
    st.write(f'The most popular artist this category is {pop_artist} with {followers} on Spotify.')
    st.write(f'If you decided to make a playlist, you can find their song "{song}" on there')
