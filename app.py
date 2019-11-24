import streamlit as st
import functions as f
import pandas as pd
import pickle
import spotipy
import spotipy.util as util
from spotipy import oauth2
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import time
import config
import numpy as np
from PIL import Image

###### model ######
model = pickle.load(open('trained_rfc_everything.pickle','rb'))

############### classes ##################
# one = 'The weird and intense'
# two = 'Everything Henry Rollins hates'
# three = 'Why am I crying in the club'
# four = 'Just let me chill, damn'
# five = 'Redbull & vodka'
# six = 'Big club energy'
# seven = 'Spacy bassy'

################### App ####################
st.title("Curation Station")
st.header("by Xristos Katsaros")
st.subheader("Generate a Spotify playlist full of independent artists based on a single song search")


song = st.text_input("Enter a song name:")
artist = st.text_input("Enter the artist name:")
genre = st.text_input("Enter a genre (optional):")
pl = st.text_input("If you're making or editing a playlist, put that here:")
username = st.text_input('Please enter your exact username to make the playlist:')


if st.button("Gimme the results"):
    f.refresh_token()
    ### check for genre ###
    if genre == "":
        genre = f.find_genre(artist,song)
    ### classify the song ###
    p_class = f.model_work(artist,song,model)
        ### get list of song ids from database ###
    id_list = f.search_db(p_class,genre)
    f.display_results(id_list)
    ### state the results ###
    # if p_class == 0:
    #     st.write(f'This song is in the "{one}" category')
    # elif p_class == 1:
    #     st.write(f'This song is in the "{two}" category ')
    # elif p_class == 2:
    #     st.write(f'This song is in the "{three}" category')
    # elif p_class == 3:
    #     st.write(f'This song is in the "{four}" category')
    # elif p_class == 4:
    #     st.write(f'This song is in the "{five}" category')
    # elif p_class == 5:
    #     st.write(f'This song is in the "{six}" category')
    # elif p_class == 6:
    #     st.write(f'This song is in the "{seven}" category')
    ### create playlist ###
    if pl != "":
        f.pl_creator(id_list,username,pl)
