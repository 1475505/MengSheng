#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
sys.path.append(r'utils')
import numpy as np
import soundfile
import librosa
from scipy import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from utilities import (read_audio, repeat_seq, scale)
import config

from models_pytorch import move_data_to_gpu, Vggish
from data_generator import DataGenerator


# In[2]:


Model = Vggish

batch_size = 64
time_steps = 128

sample_rate = config.sample_rate
window_size = config.window_size
overlap = config.overlap
mel_bins = config.mel_bins


# ## 1.提取音频特征

# ### 参数：音频位置，截取位置

# In[3]:


def calculate_logmel(audio_path, sample_rate, extractor, audio_cuts):

    (audio, _) = read_audio(audio_path, target_fs=sample_rate)

    [cut1,cut2] = audio_cuts.split(',')
    audio = audio[int(float(cut1)*sample_rate):int(float(cut2)*sample_rate)]

    audio = audio / np.max(np.abs(audio))

    feature = extractor.transform(audio)

    return feature

class LogMelExtractor():
    def __init__(self, sample_rate, window_size, overlap, mel_bins):
        
        self.window_size = window_size
        self.overlap = overlap
        self.ham_win = np.hamming(window_size)
        
        self.melW = librosa.filters.mel(sr=sample_rate, 
                                        n_fft=window_size, 
                                        n_mels=mel_bins, 
                                        fmin=50., 
                                        fmax=sample_rate // 2).T
    
    def transform(self, audio):
    
        ham_win = self.ham_win
        window_size = self.window_size
        overlap = self.overlap
    
        [f, t, x] = signal.spectral.spectrogram(
                        audio, 
                        window=ham_win,
                        nperseg=window_size, 
                        noverlap=overlap, 
                        detrend=False, 
                        return_onesided=True, 
                        mode='magnitude') 
        x = x.T
            
        x = np.dot(x, self.melW)
        x = np.log(x + 1e-8)
        x = x.astype(np.float32)
        
        return x
            
extractor = LogMelExtractor(sample_rate=sample_rate,
                            window_size=window_size,
                            overlap=overlap,
                            mel_bins=mel_bins)

feature = calculate_logmel("E:/毕设/乐器音频/三弦T0289/T0289A2-1c2-1.wav", sample_rate, extractor, "74.1,76.1")

print(feature.shape)


# ## 2.特征识别

# In[4]:


# 归一化
mean_ = np.load('mean_.npy')
std_  = np.load('std_.npy')

# 数据调整
audio_input = repeat_seq(scale(feature, mean_, std_), time_steps)
audio_input = audio_input[np.newaxis]

num_classes = len(config.labels)
model = Model(num_classes)
        
checkpoint = torch.load("md_3000_iters.tar")
model.load_state_dict(checkpoint['state_dict'])

model.cuda()
model.eval()

x_ = move_data_to_gpu(audio_input, 1)
y_ = model(x_)
y_ = y_.data.cpu().numpy()

res = np.argsort(y_)
# 预测第一
print(config.labels[np.argmax(y_)])
# 预测前三
print(config.labels[res[0][-1]], config.labels[res[0][-2]], config.labels[res[0][-3]])


# # 3.总函数

# ### （1）直接调用get_audio_result函数，参数为音频位置，音频切割点
# ### （2）最好放在一个.py文件中，直接新建拷贝即可
# ### （3）在服务器上运行时，要注意识别时间和运行环境
# ### （4）最主要的环境是torch==1.1.0  librosa==0.6.3

# In[6]:


import os
import sys
sys.path.append(r'utils')
import numpy as np
import soundfile
import librosa
from scipy import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from utilities import (read_audio, repeat_seq, scale)
import config

from models_pytorch import move_data_to_gpu, Vggish
from data_generator import DataGenerator

Model = Vggish

batch_size = 64
time_steps = 128

sample_rate = config.sample_rate
window_size = config.window_size
overlap = config.overlap
mel_bins = config.mel_bins

# 乐器种类
num_classes = len(config.labels)
model = Model(num_classes)

# 读取模型
checkpoint = torch.load("md_3000_iters.tar")
model.load_state_dict(checkpoint['state_dict'])

# 注意环境
model.cuda()
model.eval()

def calculate_logmel(audio_path, sample_rate, extractor, audio_cuts):

    (audio, _) = read_audio(audio_path, target_fs=sample_rate)
    
    if audio_cuts is not None:
        [cut1,cut2] = audio_cuts.split(',')
        audio = audio[int(float(cut1)*sample_rate):int(float(cut2)*sample_rate)]

    audio = audio / np.max(np.abs(audio))

    feature = extractor.transform(audio)

    return feature

class LogMelExtractor():
    def __init__(self, sample_rate, window_size, overlap, mel_bins):
        
        self.window_size = window_size
        self.overlap = overlap
        self.ham_win = np.hamming(window_size)
        
        self.melW = librosa.filters.mel(sr=sample_rate, 
                                        n_fft=window_size, 
                                        n_mels=mel_bins, 
                                        fmin=50., 
                                        fmax=sample_rate // 2).T
    
    def transform(self, audio):
    
        ham_win = self.ham_win
        window_size = self.window_size
        overlap = self.overlap
    
        [f, t, x] = signal.spectral.spectrogram(
                        audio, 
                        window=ham_win,
                        nperseg=window_size, 
                        noverlap=overlap, 
                        detrend=False, 
                        return_onesided=True, 
                        mode='magnitude') 
        x = x.T
            
        x = np.dot(x, self.melW)
        x = np.log(x + 1e-8)
        x = x.astype(np.float32)
        
        return x
            
extractor = LogMelExtractor(sample_rate=sample_rate,
                            window_size=window_size,
                            overlap=overlap,
                            mel_bins=mel_bins)


def get_audio_result(name, cut = None):
    # 特征
    feature = calculate_logmel(name, sample_rate, extractor, cut)
    
    # 归一化
    mean_ = np.load('mean_.npy')
    std_  = np.load('std_.npy')

    # 数据调整
    audio_input = repeat_seq(scale(feature, mean_, std_), time_steps)
    audio_input = audio_input[np.newaxis]

    x_ = move_data_to_gpu(audio_input, 1)
    y_ = model(x_)
    y_ = y_.data.cpu().numpy()

    res = np.argsort(y_)
    # 预测第一
    print(config.labels[np.argmax(y_)])
    # 预测前三
    print(config.labels[res[0][-1]], config.labels[res[0][-2]], config.labels[res[0][-3]])
    
    # 返回值可自己调整
    return config.labels[np.argmax(y_)]

# # 4.所有乐器种类

# In[6]:


config.labels


# In[ ]:




