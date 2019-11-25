import streamlit as st
import functions as f
import pickle

###### model ######
model = pickle.load(open('trained_rfc_electronic.pickle','rb'))

def main():
    ################### App ####################
    st.title("Curation Station (alpha)")
    st.header("by Xristos Katsaros")
    st.subheader('''Generate a Spotify playlist full of independent releases based on a single song search. This is is part of an
                    ongoing project. In its current state, the app only works with electronic music and its broad range of sub-genres.''')

    st.header("Required")
    song = st.text_input("Enter a song name:")
    artist = st.text_input("Enter the artist name:")
    st.header("Optional")
    genre = st.text_input("Enter a genre:")
    pl = st.text_input("Name of the new or existing playlist:")
    username = st.text_input('Enter your exact username:')


    if st.button("gimme those sweet trax ԅ(≖‿≖ԅ)"):
        f.refresh_token()
        ### check for genre ###
        if genre == "":
            genre = f.find_genre(artist,song)
        ### classify the song ###
        p_class = f.model_work(artist,song,model)
            ### get list of song ids from database ###
        id_list = f.search_db(p_class,genre)
        f.display_results(id_list,genre,p_class)
        ### create playlist ###
        if pl != "":
            f.pl_creator(id_list,username,pl)
if __name__ == '__main__':
    main()
