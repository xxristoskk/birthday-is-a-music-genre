import streamlit as st
import pandas as pd
import pickle
import spotipy
import spotipy.util as util
# from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

################### Spotify autho ######################
scope = 'playlist-modify-public'
token = util.prompt_for_user_token(config.username,
                                   scope,
                                   client_id=config.ClientID,
                                   client_secret=config.ClientSecret,
                                   redirect_uri='http://localhost/')

################## functions ###############
def find_song(artist,song):
    try:
        id_ = f.sp.search(q=f'artist:{artist} track:{song}',type='track')['tracks']['items'][0]['id']
        return id_
    except:
        print("Couldn't find what you're looking for.")

def get_features(id_):
    return f.sp.audio_features(id_)

def model_work()
################### App ####################

st.title("This is the title")
st.header("a wonkky data science project by Xristos Katsaros")

if st.button("Login/Refresh"):
    sp = spotipy.Spotify(auth=token)

song = st.text_input("Enter the song name", 'song')
artist = st.text_input("Enter the artist name",'artist')
if st.button("gimme some songs"):
    id_ = find_song(artist,song)
    st.write(get_features(id_))
