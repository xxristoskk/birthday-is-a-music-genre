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

###### load local dataset #####
data = pickle.load(open('everything_db.pickle','rb'))
data.reset_index(drop=True,inplace=True)
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
###### model ######
model = pickle.load(open('trained_rfc.pickle','rb'))
############### classes ##################
one = 'The weird and intense'
two = 'Everything Henry Rollins hates'
three = 'Why am I crying in the club'
four = 'Just let me chill, damn'
five = 'Redbull & vodka'
six = 'Big club energy'
seven = 'Spacy bassy'
################### App ####################
st.title("Genre Explorer")
st.header("by Xristos Katsaros")
st.subheader("Generate a category for a song and a list of others in the same category")

song = st.text_input("Enter a song name","bartier cardi")
artist = st.text_input("Enter the artist name",'cardi b')
# pl = st.text_input("Enter the name of your playlist")

if st.button("Generate the category"):
    f.refresh_token()
    p_class = f.model_work(artist,song,model)
    # id_list = f.search_db(data,p_class)
    if p_class == 0:
        st.write(f'This song is categorized as "{one}"')
    elif p_class == 1:
        st.write(f'This song is categorized as "{two}"')
    elif p_class == 2:
        st.write(f'This song is categorized as "{three}"')
    elif p_class == 3:
        st.write(f'This song is categorized as "{four}"')
    elif p_class == 4:
        st.write(f'This song is categorized as "{five}"')
    elif p_class == 5:
        st.write(f'This song is categorized as "{six}"')
    elif p_class == 6:
        st.write(f'This song is categorized as "{seven}"')
    # f.pl_creator(id_list,config.username,pl)
