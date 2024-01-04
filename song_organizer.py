import urllib.request
import requests
import pandas as pd
import numpy as np

np.random.seed(42)
likes = pd.read_csv("./ml_likes.csv")
dislikes = pd.read_csv("./ml_dislikes.csv")
likes_links = likes.loc[:, "Track Preview URL"]
dislikes_links = dislikes.loc[:, "Track Preview URL"]
likes_size = likes_links.shape[0]
dislikes_size = dislikes_links.shape[0]
likes_dist = np.random.randint(0, likes_size, (likes_size))
dislikes_dist = np.random.randint(0, dislikes_size, (dislikes_size))
likes_train_ind = likes_dist[0:likes_size//2]
likes_test_ind = likes_dist[likes_size//2:likes_size-1]
dislikes_train_ind = dislikes_dist[0:likes_size//2]
dislikes_test_ind = dislikes_dist[likes_size//2:likes_size-1]

temp1 = []
temp2 = []
for i in likes_train_ind:
    temp1.append(likes_links[i])
for j in dislikes_train_ind:
    temp2.append(dislikes_links[j])
likes_train_data = np.array(temp1)
dislikes_train_data = np.array(temp2)
temp1 = []
temp2 = []
for i in likes_test_ind:
    temp1.append(likes_links[i])
for j in dislikes_test_ind:
    temp2.append(dislikes_links[j])
likes_test_data = np.array(temp1)
dislikes_test_data = np.array(temp2)

#Get training data for likes
for k in range(len(likes_train_data)):
    try:
        song_preview = requests.get(likes_train_data[k])
        file_path = "liked_songs/train_data/sample_" + str(k) + ".wav"
        with open(file_path, 'wb') as f:
            f.write(song_preview.content)
    except:
        print("song had no preview link")

#get test data for likes
for k in range(len(likes_test_data)):
    try:
        song_preview = requests.get(likes_test_data[k])
        file_path = "liked_songs/test_data/sample_" + str(k) + ".wav"
        with open(file_path, 'wb') as f:
            f.write(song_preview.content)
    except:
        print("song (" + str(k) + ") had no preview link")

#get training data for dislikes
for k in range(len(dislikes_train_data)):
    try:
        song_preview = requests.get(dislikes_train_data[k])
        file_path = "disliked_songs/train_data/sample_" + str(k) + ".wav"
        with open(file_path, 'wb') as f:
            f.write(song_preview.content)
    except:
        print("song (" + str(k) + ") had no preview link")

#get training data for dislikes
for k in range(len(dislikes_test_data)):
    try:
        song_preview = requests.get(dislikes_test_data[k])
        file_path = "disliked_songs/test_data/sample_" + str(k) + ".wav"
        with open(file_path, 'wb') as f:
            f.write(song_preview.content)
    except:
        print("song (" + str(k) + ") had no preview link")


