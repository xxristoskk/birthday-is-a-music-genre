## Last updated json Oct 6th. 12, 2019
import json
import config
import functions as f
from tqdm import tqdm

##### establish classes for dashboard #####
# class User():
#     def __init__(self,username,password):
#         self.username = username
#         self.password = password
#     pass
#
# class Curator():
#     def __init__(self,user)


## Helper functions
## Takes in data and creates a list of all genres
def get_genres(data):
    lst = []
    ## List all the tags/gnres you don't want in your playlist
    ## Removing general genres or tags like Electronic or Album will yield more curated results
    rm_list = ['Album','Single','EP','Various Artists']
    for x in data:
        lst.append(x['genres'])
    return lst

## Takes in dictionary of releases and builds a dictionary of genres
## Each genre is a key and its values are another dictionary of genres the key is paired with (neighbors)
## The value of the neighbors is equal to the number of times the two genres were paired together
def genre_dict_builder(data):
    genre_list = get_genres(data)
    genre_dict = {}
    for genre in tqdm(genre_list):
        for x in range(len(genre)):
            if genre[x-1] not in genre_dict:
                genre_dict[genre[x-1]] = {}
            if genre[x] not in genre_dict[genre[x-1]]:
                genre_dict[genre[x-1]][genre[x]] = 0
            genre_dict[genre[x-1]][genre[x]] += 1
    return genre_dict

# Takes in a dictionary of genres and a list of genres
# Finds the closest genre neighbors and adds them to a new list (tuned)
# The tuned list is checked for duplicates and a final list is returned
def genre_tuner(dictionary, genres):
    genre_tuples = []
    tuned = []
    final = []
    for genre in genres:
        if genre not in dictionary.keys():
            print('Nothing found')
        else:
            genre_tuples.append(sorted(dictionary[genre].items(),key=lambda tup: tup[1],reverse=True)[:10]) # sorts tuples in decsending order
    for items in genre_tuples:
        i=50
        for tup in items:
            if tup[1] in range(0,i) and tup[0] not in genres:
                i = tup[1]
                tuned.append(tup[0])
    #Checks for duplicates
    for n in tuned:
        if n not in final:
            final.append(n)
    print(genre_tuples)
    return final

# Takes in list of releases as dictionaries, along with a list of genres
# Finds the neighboring genres and if both the neighbors and listed genres are in the releases genre, it is appended to new list
def curated_data(data, genres):
    genres = [i.title() for i in genres]
    neighbors = genre_tuner(genre_dict_builder(data),genres)
    new = []
    for release in data:
        for neighbor in neighbors:
            for genre in genres:
                if neighbor not in release['genres']:
                    continue
                elif neighbor in release['genres'] and genre in release['genres']:
                    new.append(release)
    new = remove_duplicates(new)
    return new

## Remove for duplicate releases
def remove_duplicates(data):
    l1 = []
    l2 = []
    for release in tqdm(data):
        if release['album'] in l1:
            continue
        else:
            l1.append(release['album'])
            l2.append(release)
    return l2
