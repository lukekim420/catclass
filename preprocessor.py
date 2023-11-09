import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# 관련 라이브러리 가져오기
import matplotlib.pyplot as plt

import os
from glob import glob

base_dir = './audioset-processing-master/output/meow'
train_folder = glob(base_dir)

train_path = []
for folder in train_folder:
    tmp = glob(folder + '/*')
    train_path += tmp

def ctft_to_peaks(carr : np.ndarray, threshold = 1, max_peak_num = 10):
    rising = np.zeros(shape=(max_peak_num, carr.shape[1]))
    falling = np.zeros(shape=(max_peak_num , carr.shape[1]))
    diff=np.diff(np.sign(carr.T-threshold), axis = 1)

    for arr, num in [(rising, 2), (falling, -2)]:
        a,b = np.where(diff==num)
        for i in range(diff.shape[0]):
            temp=b[a==i]
            minlen = min(max_peak_num, len(temp))
            arr[:minlen,i]=temp[:minlen]

    return (rising, falling)


for vidno in range(len(train_path)):
    y, sr = librosa.load(train_path[vidno])

    S_full, phase = librosa.magphase(librosa.stft(y))



    # We'll compare frames using cosine similarity, and aggregate similar frames
    # by taking their (per-frequency) median value.
    #
    # To avoid being biased by local continuity, we constrain similar frames to be
    # separated by at least 2 seconds.
    #
    # This suppresses sparse/non-repetetitive deviations from the average spectrum,
    # and works well to discard vocal elements.

    S_filter = librosa.decompose.nn_filter(S_full,
                                        aggregate=np.median,
                                        metric='cosine',
                                        width=int(librosa.time_to_frames(2, sr=sr)))

    # The output of the filter shouldn't be greater than the input
    # if we assume signals are additive.  Taking the pointwise minimium
    # with the input spectrum forces this.
    S_filter = np.minimum(S_full, S_filter)
    # We can also use a margin to reduce bleed between the vocals and instrumentation masks.
    # Note: the margins need not be equal for foreground and background separation
    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                margin_i * (S_full - S_filter),
                                power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                margin_v * S_filter,
                                power=power)

    # Once we have the masks, simply multiply them with the input spectrum
    # to separate the components
    S_foreground = mask_v * S_full
    S_background = mask_i * S_full
    # sphinx_gallery_thumbnail_number = 2


    rising, falling = ctft_to_peaks(S_foreground, threshold=0.1, max_peak_num=5)

    x=np.arange(0, rising.shape[1],1/1).astype(int)


    #의미있는 부분만 잘라내기
    cnt_mat = (rising > 1e-5).sum(axis=0)
    thres = 5 - 0.5
    cutpoint_rising = np.where(np.diff(np.sign(cnt_mat-thres))==2)[0]
    cutpoint_falling = np.where(np.diff(np.sign(cnt_mat-thres))==-2)[0]+1
    if(cnt_mat[0] > thres):
        cutpoint_rising = np.insert(cutpoint_rising,0,0)

    if len(cutpoint_rising)!=len(cutpoint_falling):
        cutpoint_falling = cutpoint_falling[:-1]

    for i in range(len(cutpoint_rising)):
        if 1.3>=(librosa.frames_to_time(cutpoint_falling[i],sr=sr)-librosa.frames_to_time(cutpoint_rising[i],sr=sr))>=0.3:
            #print(cutpoint_rising[i], cutpoint_falling[i])
            print(("ffmpeg -y -ss " + str(librosa.frames_to_time(cutpoint_rising[i],sr=sr)) + " -t " + str(librosa.frames_to_time(cutpoint_falling[i],sr=sr)-librosa.frames_to_time(cutpoint_rising[i],sr=sr)) + " -i " +str(train_path[vidno])+" ./dataset/meow/"+str(train_path[vidno].split('/')[-1].split(".")[0])+'_'+str(i)+".wav"))
            os.system(("ffmpeg -y -ss " + str(librosa.frames_to_time(cutpoint_rising[i],sr=sr)) + " -t " + str(librosa.frames_to_time(cutpoint_falling[i],sr=sr)-librosa.frames_to_time(cutpoint_rising[i],sr=sr)) + " -i " +str(train_path[vidno])+" ./dataset/meow/"+str(train_path[vidno].split('/')[-1].split(".")[0])+'_'+str(i)+".wav"))