import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from PIL import Image
import math

# def Base_cnn(input_shape):
#     Ini=Input(input_shape)
        
#     # 第一个卷积部分
#     x = Conv2D(32,(3,3),activation = 'relu',padding = 'same',bias_regularizer=regularizers.l2(0.01))(Ini)
#     x = MaxPooling2D((2,2), padding='valid')(x)
        
#     # 第二个卷积部分
#     x = Conv2D(64,(3,3),activation = 'relu',padding = 'same',bias_regularizer=regularizers.l2(0.01))(x)
#     x = MaxPooling2D((2,2),padding='valid')(x)
    
#     # 第二个卷积部分
#     x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',bias_regularizer=regularizers.l2(0.01))(x)
#     x = MaxPooling2D((2,2),padding='valid')(x)
    
#     x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',bias_regularizer=regularizers.l2(0.01))(x)
#     x = MaxPooling2D((2,2),padding='valid')(x)
    
      
#     x = Flatten()(x)
#     x = Dropout(0.1)(x)
#     x = Dense(256, activation='relu')(x)
#     x = Dropout(0.1)(x)
#     x = Dense(128, activation='relu')(x)
#     out = Dense(64, activation='relu')(x)

#     return Model(Ini,out)

def eca_block(inputs, b=1, gama=2):
    
    # 输入特征图的通道数
    in_channel = inputs.shape[-1]
    
    # 根据公式计算自适应卷积核大小
    kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
    
    # 如果卷积核大小是偶数，就使用它
    if kernel_size % 2:
        kernel_size = kernel_size
    
    # 如果卷积核大小是奇数就变成偶数
    else:
        kernel_size = kernel_size + 1
    
    # [h,w,c]==>[None,c] 全局平均池化
    x = layers.GlobalAveragePooling2D()(inputs)
    
    # [None,c]==>[c,1]
    x = layers.Reshape(target_shape=(in_channel, 1))(x)
    
    # [c,1]==>[c,1]
    x = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)
    
    # sigmoid激活
    x = tf.nn.sigmoid(x)
    
    # [c,1]==>[1,1,c]
    x = layers.Reshape((1,1,in_channel))(x)
    
    # 结果和输入相乘
    outputs = layers.multiply([inputs, x])
    
    return outputs




def Base_cnn(input_shape):
    Ini=Input(input_shape)
        
    # 第一个卷积部分
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same')(Ini)
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same')(x)
    x = MaxPooling2D((2,2), padding='valid')(x)
        
    # 第二个卷积部分
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same')(x)
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same')(x)
    x = MaxPooling2D((2,2),padding='valid')(x)
    
#     # 第二个卷积部分
#     x = Conv2D(256,(3,3),activation = 'relu',padding = 'same')(x)
#     x = Conv2D(256,(3,3),activation = 'relu',padding = 'same')(x)
#     x = Conv2D(256,(3,3),activation = 'relu',padding = 'same')(x)
#     x = MaxPooling2D((2,2),padding='valid')(x)
    
#     x = Conv2D(512,(3,3),activation = 'relu',padding = 'same')(x)
#     x = Conv2D(512,(3,3),activation = 'relu',padding = 'same')(x)
#     x = Conv2D(512,(3,3),activation = 'relu',padding = 'same')(x)
    x = MaxPooling2D((2,2),padding='valid')(x)
    
    
      
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(32, activation='relu')(x)
    
    return Model(Ini,out)

def Base_cnn_ec(input_shape):
    Ini=Input(input_shape)
        
    # 第一个卷积部分
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same')(Ini)
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same')(x)
    x = MaxPooling2D((2,2), padding='valid')(x)
        
    # 第二个卷积部分
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same')(x)
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same')(x)
    x = MaxPooling2D((2,2),padding='valid')(x)
    
#     # 第二个卷积部分
#     x = Conv2D(256,(3,3),activation = 'relu',padding = 'same')(x)
#     x = Conv2D(256,(3,3),activation = 'relu',padding = 'same')(x)
#     x = Conv2D(256,(3,3),activation = 'relu',padding = 'same')(x)
#     x = MaxPooling2D((2,2),padding='valid')(x)
    
#     x = Conv2D(512,(3,3),activation = 'relu',padding = 'same')(x)
#     x = Conv2D(512,(3,3),activation = 'relu',padding = 'same')(x)
#     x = Conv2D(512,(3,3),activation = 'relu',padding = 'same')(x)
#     x = MaxPooling2D((2,2),padding='valid')(x)
    x = eca_block(x)
    
      
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(32, activation='relu')(x)
    
    return Model(Ini,out)