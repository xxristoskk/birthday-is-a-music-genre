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
st.title("Genre Explorer")
st.header("by Xristos Katsaros")
st.subheader("Generate a category for a song and a list of others in the same category")

song = st.text_input("Enter a song name:")
artist = st.text_input("Enter the artist name:")
genre = st.text_input("Enter a genre (optional):")
pl = st.text_input("Enter a name for your playlist or an existing playlist:")



if st.button("Show me what you got"):
    f.refresh_token()
    ### check for genre ###
    if genre == "":
        genre = f.find_genre(artist,song)
    ### classify the song ###
    p_class = f.model_work(artist,song,model)

    ### get list of song ids from database ###
    id_list = f.search_db(p_class,genre)

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
    f.pl_creator(id_list,config.username,pl)
