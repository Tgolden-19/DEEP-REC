import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import pydub
import numpy as np

def load_and_split_sample(filepath):
    #load audio via tensorflow
    audio = pydub.AudioSegment.from_mp3(filepath)
    audio = np.array(audio.get_array_of_samples())
    # print(audio.shape)
    tensor = tf.convert_to_tensor(audio)
    # audio = tf.io.read_file(filepath)

    # tensor = tf.squeeze(tensor, axis=[-1])
    # audio_size = audio.shape[0]
    # pt_1 = audio[:audio_size//3]
    # pt_2 = audio[audio_size//3:(audio_size//3)*2]
    # pt_3 = audio[(audio_size//3)*2:]
    # pt_1_tensor = tf.squeeze(pt_1, axis=[-1])
    # pt_2_tensor = tf.squeeze(pt_2, axis=[-1])
    # pt_3_tensor = tf.squeeze(pt_3, axis=[-1])
    # return pt_1_tensor, pt_2_tensor, pt_3_tensor
    return audio


test_file = os.path.join('liked_songs', 'train_data','sample_0.wav')
# print(os.path.exists(test_file))
tensor = load_and_split_sample(test_file)
def split_clips(dir_path):
    # clip1, clip2, clip3 = load_and_split_sample(test_file)
    clip = load_and_split_sample(test_file)
    slices = tf.split(clip, 6)
    # slices = tf.stack(slices)
    tensor = tf.cast(slices, tf.float32) / 32768.0
    # tensor1 = tf.cast(clip1, tf.float32) / 32768.0
    # tensor2 = tf.cast(clip2, tf.float32) / 32768.0
    # tensor3 = tf.cast(clip3, tf.float32) / 32768.0
    #plot waveforms to verify loading
    # plt.figure()
    # plt.plot(tensor1.numpy())
    # plt.plot(tensor2.numpy())
    # plt.plot(tensor3.numpy())
    # plt.show()
    # print(tensor)

    # spectrogram1 = tfio.audio.spectrogram(
    #     tensor1, nfft=512, window=512, stride=256)
    # spectrogram2 = tfio.audio.spectrogram(
    #     tensor2, nfft=512, window=512, stride=256)
    # spectrogram3 = tfio.audio.spectrogram(
    #     tensor3, nfft=512, window=512, stride=256)



    # plt.imshow(tf.math.log(spectrogram2).numpy())

    #old return
    # return spectrogram1, spectrogram2, spectrogram3, label
    #new return
    return tensor


def to_spectrogram(tensor):
    spectrogram = tfio.audio.spectrogram(
        tensor, nfft=512, window=512, stride=256)
    return spectrogram

def prep_datasets():
    #create paths to training data
    likes_path = os.path.join("/content/drive/MyDrive/practicum_proj/data/liked_songs", "train_data")
    dislikes_path = os.path.join("/content/drive/MyDrive/practicum_proj/data/disliked_songs", "train_data")
    # print(os.path.exists(likes_path+"/sample_0.mp3"))
    # print(os.path.exists(dislikes_path))

    #load relevant samples
    like_set = split_clips(likes_path+"/sample_0.mp3")
    print(like_set)
    like_files = os.listdir(likes_path)
    # like_data = tf.data.Dataset.list_files(likes_path+"/*.mp3")
    # dislike_data = tf.data.Dataset.list_files(dislikes_path+"/*.mp3")
    for x in range(2,len(like_files)):
      file_name = like_files[x]
      # print(str(x) + ' : ' + str(len(like_files)))
      temp_path = os.path.join(likes_path, file_name)
      # print(temp_path)
      # print(os.path.exists(temp_path))
      temp_tensor = split_clips(temp_path)
      like_set = tf.concat([like_set, temp_tensor], 0)

    print('done likes train')
    print(like_set)

    dislike_set = split_clips(dislikes_path+"/sample_0.mp3")
    # print(dislike_set)
    dislike_files = os.listdir(dislikes_path)
    # like_data = tf.data.Dataset.list_files(likes_path+"/*.mp3")
    # dislike_data = tf.data.Dataset.list_files(dislikes_path+"/*.mp3")
    for x in range(2,len(like_files)):
      # print(x)
      file_name = dislike_files[x]
      temp_path = os.path.join(dislikes_path, file_name)
      # print(temp_path)
      temp_tensor = split_clips(temp_path)
      dislike_set = tf.concat([dislike_set, temp_tensor], 0)
    print('done dislikes train')
    print(dislike_set)

    #convert to datasets
    like_data = tf.data.Dataset.from_tensor_slices(like_set)
    dislike_data = tf.data.Dataset.from_tensor_slices(dislike_set)
    # like_data = like_data.map(split_clips)
    # dislike_data = dislike_data.map(split_clips)
    # print(like_data)
    # #-----------------------
    # dataset_to_numpy = list(like_data.as_numpy_iterator())
    # shape = tf.shape(dataset_to_numpy)
    # print(shape)
    # #---------------------
    # dislike_data = tf.concat(dislike_data, 2)
    # like_data = tf.concat(like_data, 2)
    like_spect = like_data.map(to_spectrogram)
    dislike_spect = dislike_data.map(to_spectrogram)

    #tensors are currently in groups of 6 and need to be flattened to 1 column of tensors
    # like_spect = tf.stack(like_spect)
    # dislike_spect = tf.stack(dislike_spect)

    # like_shape = tf.shape(like_spect)
    # dislike_shape = tf.shape(dislike_spect)
    # like_spect = tf.reshape(like_spect, [])
    # dislike_spect = tf.reshape([])

    likes = tf.data.Dataset.zip((like_spect, tf.data.Dataset.from_tensor_slices(tf.ones(len(like_data)))))
    dislikes = tf.data.Dataset.zip((dislike_spect, tf.data.Dataset.from_tensor_slices(tf.zeros(len(dislike_data)))))

    train_data = likes.concatenate(dislikes)

    train_data = train_data.cache().shuffle(buffer_size=2500)
    train_data = train_data.batch(32)
    train_data = train_data.prefetch(4)


    likes_path = os.path.join("/content/drive/MyDrive/practicum_proj/data/liked_songs", "test_data")
    dislikes_path = os.path.join("/content/drive/MyDrive/practicum_proj/data/disliked_songs", "test_data")
    # print(os.path.exists(likes_path))
    # print(os.path.exists(dislikes_path))


        #create paths to training data
    likes_path = os.path.join("/content/drive/MyDrive/practicum_proj/data/liked_songs", "train_data")
    dislikes_path = os.path.join("/content/drive/MyDrive/practicum_proj/data/disliked_songs", "train_data")
    # print(os.path.exists(likes_path+"/sample_0.mp3"))
    # print(os.path.exists(dislikes_path))

    #load relevant samples
    like_set = split_clips(likes_path+"/sample_0.mp3")
    print(like_set)
    like_files = os.listdir(likes_path)
    # like_data = tf.data.Dataset.list_files(likes_path+"/*.mp3")
    # dislike_data = tf.data.Dataset.list_files(dislikes_path+"/*.mp3")
    for x in range(2,len(like_files)):
      file_name = like_files[x]
      # print(file_name)
      temp_path = os.path.join(likes_path, file_name)
      # print(temp_path)
      # print(os.path.exists(temp_path))
      temp_tensor = split_clips(temp_path)
      likeset = tf.concat([like_set, temp_tensor], 0)
    print('done likes test')
    print(like_set)

    dislike_set = split_clips(dislikes_path+"/sample_0.mp3")
    # print(dislike_set)
    dislike_files = os.listdir(dislikes_path)
    # like_data = tf.data.Dataset.list_files(likes_path+"/*.mp3")
    # dislike_data = tf.data.Dataset.list_files(dislikes_path+"/*.mp3")
    for x in range(2,len(like_files)):
      file_name = dislike_files[x]
      temp_path = os.path.join(dislikes_path, file_name)
      # print(temp_path)
      temp_tensor = split_clips(temp_path)
      dislikeset = tf.concat([dislike_set, temp_tensor], 0)
    print('done dislikes test')
    print(dislike_set)

    #convert to datasets
    like_data = tf.data.Dataset.from_tensor_slices(like_set)
    dislike_data = tf.data.Dataset.from_tensor_slices(dislike_set)


    # #create relevant test datasets
    # like_data = tf.data.Dataset.list_files(likes_path+"/*.mp3")
    # dislike_data = tf.data.Dataset.list_files(dislikes_path+"/*.mp3")
    # like_data = like_data.map(split_clips)
    # dislike_data = dislike_data.map(split_clips)
    # # #-----------------------
    # # dataset_to_numpy = list(like_data.as_numpy_iterator())
    # # shape = tf.shape(dataset_to_numpy)
    # # print(shape)
    # # #---------------------
    # # dislike_data = tf.concat(dislike_data, 2)
    # # like_data = tf.concat(like_data, 2)
    # like_spect = like_data.map(to_spectrogram)
    # dislike_spect = dislike_data.map(to_spectrogram)

    # # #tensors are currently in groups of 6 and need to be flattened to 1 column of tensors
    # # like_spect = tf.stack(like_spect)
    # # dislike_spect = tf.stack(dislike_spect)

    # # like_shape = tf.shape(like_spect)
    # # dislike_shape = tf.shape(dislike_spect)
    # # like_spect = tf.reshape(like_spect, [])
    # # dislike_spect = tf.reshape([])

    likes = tf.data.Dataset.zip((like_spect, tf.data.Dataset.from_tensor_slices(tf.ones(len(like_data)))))
    dislikes = tf.data.Dataset.zip((dislike_spect, tf.data.Dataset.from_tensor_slices(tf.zeros(len(dislike_data)))))

    test_data = likes.concatenate(dislikes)
    test_data = test_data.cache().shuffle(buffer_size=2500).batch(32).prefetch(4)
    return train_data, test_data

train_data, test_data = prep_datasets()