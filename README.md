# birthday-is-a-music-genre
Sifting through the noise in the electronic music on Spotify

## What is this?
Spotify has over 1,500 genre tags. One of them being "birthday" and another one being "skinhead oi." Why, or how, the latter is in their list is a subject of a completely different project. The purpose of this project is to create a music discovery tool that takes in a song and generates a playlist of x amount of songs that share its features. What makes this project standout from others is that the generated playlist will consist of artists in a database of independent and DIY artists which I have been in the process of developing.

## The approach
* Scraped artist names and genre info from two blogs and the names of all artists currently on bandcamp.com
* Searched for all of those artists, using the Spotify API, to confirm their music is available on that platform
* Collected audio features provided by the API of those artists' top songs
* Performed K-Means clustering and labeled the clusters according to their most prominent features
* Trained a Random Forest Classifier to identify which cluster a song would belong

## The results
* Although some of the clusters reflected their prominent features, the features provided by Spotify are too subjective for the purpose of this project
* The classifier performed great with 93% accuracy, but because of the issue raised above, the content of what was being classified were mislabled 

## The Future
* I'd like to do more feature engineering and perform my own audio analysis with more objective metrics--find patterns in dynamics and cadence
* Put together the database of DIY artists
* Develop a front end (in the curation station repo) for a complete web application that can be used by anyone
