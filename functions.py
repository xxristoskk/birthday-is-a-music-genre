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

######### NOTE ##########
""" a few of these functions are redundant and can be consolidated. i am keeping them as is right now
    for the sake of time, but will begin refactoring once the beta version is complete """

scope = 'playlist-modify-public'
# scope = 'user-top-read'
# auth method one -- this method doesn't ask you for authorization every time the script runs
### but it also doesn't work with the refresh token
###### will use this method again for final product

token = util.prompt_for_user_token(config.username,
                                   scope,
                                   client_id=config.ClientID,
                                   client_secret=config.ClientSecret,
                                   redirect_uri='http://localhost/')
sp = spotipy.Spotify(auth=token)

##################### oauth2 for token refreshing ###########################
####### is_token_expired attribute isn't working for spotipy so declaring it from the latest version oauth2.py on the github
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

#### function that checks the status of the access token ####
def refresh_token():
    global token_info, sp
    if is_token_expired(token_info):
        token_info = oauth.refresh_access_token(token_info['refresh_token'])
        token = token_info['access_token']
        sp = spotipy.Spotify(auth=token)

###### cleaning the genre list made from the keys of no/gb genre dict #####
def clean_list(lst):
    clean = []
    for genre in lst:
        genre = genre.lower()
        if " / " in genre:
            clean.append(genre.split(" / ")[1])
            clean.append(genre.split(" / ")[0])
        else:
            clean.append(genre)
    return clean

##### cleaning the df of noise #####
def clean_df(df):
    garbage = ['ep','album','single','compilation','various artists','remixes','promo']
    for col in df.columns:
        if col in garbage:
            df.drop(columns=col,inplace=True)
    for ind in df.index:
        if ind in garbage:
            df.drop(index=ind, inplace=True)
    return

##### takes in a list of artists and returns their group of genres
def find_genres(artist_list):
    genres = []
    for artist in tqdm(artist_list):
        try:
            r = f.find_artist(artist)
            if len(r['artists']['items']) > 1:
                for results in r['artists']['items']:
                    if len(results['genres']) < 1:
                        continue
                    else:
                        genres.append(results['genres'])
        except:
            print('something dumb happened')
            continue
    return genres

##### Function takes in a list of genres, saves the top ten artists and their top ten songs, saves a list of dictionaries
def spotify_genre_dict(genres):
    case_list = []
    dictionary = {}
    for genre in tqdm(genres):
        open('g_feat_tup.pickle','ab')
        artist_results = sp.search(q=f'genre:{genre}',type='artist')['artists']['items']
        artists = [(x['name'],x['id']) for x in artist_results]
        related_genres = [x['genres'] for x in artist_results]
        top_trax = [get_top_tracks(id) for artist,id in artists]
        dictionary = {'genre': genre,
                      'related_genres': related_genres,
                      'artists': [x[0] for x in artists],
                      'artist_ids': [x[1] for x in artists],
                      'top_trax': top_trax
                      }
        tups = (genre, sp.audio_features([x[1] for x in artists]))
        case_list.append(tups)
        refresh_token()
    pickle.dump(case_list,open('g_feat_tup.pickle','wb'))
    return

def get_top_tracks(artist_id):
    return [x['id'] for x in sp.artist_top_tracks(artist_id)['tracks']]

###### cleaning the genre list made from the keys of no/gb genre dict #####
def clean_list(lst):
    clean = []
    for genre in lst:
        genre = genre.lower()
        if " / " in genre:
            clean.append(genre.split(" / ")[1])
            clean.append(genre.split(" / ")[0])
        else:
            clean.append(genre)
    return clean

###### takes in a list of dictionaries and returns the artist ids
def grab_artist_ids(lod):
    a_ids = []
    for genre in tqdm(full_top_artists):
        if len(genre['artists']) < 1:
            continue
        else:
            for id in genre['artist_ids']:
                a_ids.append(id)
    return(a_ids)

##### takes in a list of bandcamp artist names and run a search to confirm they are on spotify
def bc_artist_search(bc_list):
    """ takes in the list of bandcamp artists and searches for them on spotify
    if they are on spotify the artist is added to a new list """
    new_list = pickle.load(open('bc_confirmed.pickle','rb')) ### all the artists confirmed to be on spotify
    for artist in tqdm(bc_list):
        f.refresh_token()
        results = f.find_artist(artist)
        if results['artists']['total'] < 1:
            continue
        elif results['artists']['total'] == 1:
            new_list.append((artist,results['artists']['items'][0]['id']))
            pickle.dump(new_list,open('bc_confirmed.pickle','wb'))
    return

#### takes in a list of bandcamp artists that are on spotify and returns a dictionary with their stats
## this will eventually replace the bc_artist_search function
def get_bc_artist_info(loa):
    case_list = []
    bc_dict = {}
    for i in tqdm(range(len(loa))):
        open('new_bc_dict.pickle','a')
        try:
            r = f.find_artist(loa[i][0])['artists']['items'][0]
            bc_dict = {'followers': r['followers']['total'],
                       'genres': r['genres'],
                       'id': r['id'],
                       'artist_name': r['name'],
                       'popularity': r['popularity']}
            case_list.append(bc_dict)
            pickle.dump(case_list,open('new_bc_dict.pickle','wb'))
        except:
            print('no longer on spotify')
    return

#### function takes in the list of tuples and creates a new column 'genre' which is = [0] and the data comes from [1] #####
def merge_dfs(list_of_df):
    list_of_genres = [x[0] for x in list_of_df if x[1][0] != None]
    starter_df = pd.DataFrame(list_of_df[0][1]).reset_index().rename(columns={'index':'genre'})
    starter_df['genre'] = list_of_genres[0]
    for i in tqdm(range(1,len(list_of_df))):
        df = pd.DataFrame(list_of_df[i][1]).reset_index().rename(columns={'index':'genre'})
        df['genre'] = list_of_df[i][0]
        starter_df = pd.concat([starter_df,df])
    return starter_df


####### function to flatten out lists of lists (i made a lot of those and working against time) ######
def flatten_lists(list_of_lists):
    return [x for y in list_of_lists for x in y]

## helper functions for pl_creator
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
def pl_creator(data, user, pl_name):
    pl_id = check_playlist(user,pl_name)
    album_ids = []
    track_ids = []
    ## Search for albums in the dictionary
    for x in data:
        try:
            results = search_album(x['album'])
            album_name = results['albums']['items'][0]['name']
        except:
            data.remove(x) ## if the release can't be found on spotify, it is removed from from the search
        if album_name == x['album']:
            album_ids.append(results['albums']['items'][0]['id'])
        else:
            continue
    ## Put all found music into the playlist
    n = len(album_ids)
    i=0
    ## Adds 100 songs at a time
    while i < range(n):
        for z in range(99):
            track_ids.append(get_track_ids(album_ids[z]))
            i+=1
        add_to_playlist(user,pl_id,track_ids[:99])
    return
