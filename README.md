# birthday-is-a-music-genre


## What is this?
Spotify has over 1,500 genre tags. The purpose of this project is to create a music discovery tool that takes in a song and generates a playlist of x amount of songs that share its audio features. What makes this project unique from others like it, is that the generated playlist will consist of artists that are also on Bandcamp.com--one of the most popular distributors for independent artists.

## The approach
* Scraped artist names and genre info from two blogs and the names of all artists currently on Bandcamp.com
* Searched for all of those artists, using the Spotify API, to confirm their music is available on that platform
* Collected audio features provided by the API of those artists' top songs
* Performed K-Means clustering and labeled the clusters according to their most prominent features
* Trained a Random Forest Classifier to identify which cluster a song would belong

Having a look at the distribution of the number of artists in each genre, I decided to focus clustering only artists in the Electronic music genre. There would be no use in trying to cluster all the genres, because the features are not unique to any of them. By clustering just one of the broader genres, the clusters could more accurately represent the subgenres. As of now, I am figuring out a way to train a model that will be able to classify all the genres.

![Distribution of artists on Bandcamp grouped by genre](https://github.com/xxristoskk/birthday-is-a-music-genre/blob/master/Visuals/bc_genre_dist.png)

## The results
* The classifier performed at 90% accuracy in classifying the clusters
* Playlists accurately capture the cluster's most prominent features which can be seen below

![Heatmap of the audio features for each cluster label](https://github.com/xxristoskk/birthday-is-a-music-genre/blob/master/Visuals/labels_md.png)

## The Future
* More feature engineering and perform my own audio analysis with more objective metrics--find patterns in dynamics and cadence
* Keep updating the database
* Develop a front end (in the curation station repo) for a complete web application that can be used by anyone

## Current status
* MongoDB collection has over 500k Bandcamp artists and the database is consistently maintained
* Streamlit app hosted on Heroku works well with electronic music
