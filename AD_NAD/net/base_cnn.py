import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input, MaxPooling1D,Dropout,Bidirectional,LSTM, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from PIL import Image
import math




def Base_cnn(input_shape):
    Ini=Input(input_shape)
        
    # 第一个卷积部分
    x = Conv1D(16,3,activation = 'relu',padding = 'same',bias_regularizer=regularizers.l2(0.01))(Ini)
    #x = Conv1D(16,3,activation = 'relu',padding = 'same',bias_regularizer=regularizers.l2(0.01))(Ini)
    x = MaxPooling1D(2, strides = 2, padding='valid')(x)
        
    # 第二个卷积部分
    x = Conv1D(32,3,activation = 'relu',padding = 'same',bias_regularizer=regularizers.l2(0.01))(x)
    #x = Conv1D(32,3,activation = 'relu',padding = 'same',bias_regularizer=regularizers.l2(0.01))(x)
    x = MaxPooling1D(2,strides = 2,padding='valid')(x)
    
    # 第二个卷积部分
    #x = Conv1D(64,3,activation = 'relu',padding = 'same',bias_regularizer=regularizers.l2(0.01))(x)
    x = Conv1D(64,3,activation = 'relu',padding = 'same',bias_regularizer=regularizers.l2(0.01))(x)
    x = MaxPooling1D(2,strides = 2,padding='valid')(x)
    
    #x = Conv1D(64,3,activation = 'relu',padding = 'same',bias_regularizer=regularizers.l2(0.01))(x)
    x = Conv1D(64,3,activation = 'relu',padding = 'same',bias_regularizer=regularizers.l2(0.01))(x)
    x = MaxPooling1D(2,strides = 2,padding='valid')(x)
    
    x = Conv1D(64,3,activation = 'relu',padding = 'same',bias_regularizer=regularizers.l2(0.01))(x)
    x = MaxPooling1D(2,strides = 2,padding='valid')(x)
    
    x = Conv1D(64,3,activation = 'relu',padding = 'same',bias_regularizer=regularizers.l2(0.01))(x)
    x = MaxPooling1D(2,strides = 2,padding='valid')(x)
   
    
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    out = Dense(16, activation='relu')(x)
    
    return Model(Ini,out)






#信道间注意力
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
    
    # [w,c]==>[None,c] 全局平均池化
    x = layers.GlobalAveragePooling1D()(inputs)
    
    # [None,c]==>[c,1]
    x = layers.Reshape(target_shape=(in_channel, 1))(x)
    
    # [c,1]==>[c,1]
    x = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)
    
    # sigmoid激活
    x = tf.nn.sigmoid(x)
    
    # [c,1]==>[1,c]
    x = layers.Reshape((1,in_channel))(x)
    
    # 结果和输入相乘
    outputs = layers.multiply([inputs, x])
    
    return outputs



#--2s---test
def Base_cnn_1(input_shape):
    Ini=Input(input_shape)
        
    # 第一个卷积部分
    x = Conv1D(64,3,activation = 'relu',padding = 'same')(Ini)
    x = MaxPooling1D(2, strides = 2, padding='valid')(x)
        
    # 第二个卷积部分
    x = Conv1D(64,3,activation = 'relu',padding = 'same')(x)
    x = MaxPooling1D(2,strides = 2,padding='valid')(x)
    
    # 第二个卷积部分
    x = Conv1D(128,3,activation = 'relu',padding = 'same')(x)
    x = MaxPooling1D(2,strides = 2,padding='valid')(x)
    
    x = Conv1D(128,3,activation = 'relu',padding = 'same')(x)
    x = MaxPooling1D(2,strides = 2,padding='valid')(x)
    
    x = Conv1D(256,3,activation = 'relu',padding = 'same')(x)
    x = Conv1D(256,3,activation = 'relu',padding = 'same')(x)
    x = MaxPooling1D(2,strides = 2,padding='valid')(x)
    
    x = Conv1D(256,3,activation = 'relu',padding = 'same')(x)
    x = Conv1D(256,3,activation = 'relu',padding = 'same')(x)
    x = MaxPooling1D(2,strides = 2,padding='valid')(x)
    
    x = Conv1D(512,3,activation = 'relu',padding = 'same')(x)
    x = Conv1D(512,3,activation = 'relu',padding = 'same')(x)
    x = Conv1D(512,3,activation = 'relu',padding = 'same')(x)
    x = MaxPooling1D(2,strides = 2,padding='valid')(x)
    
    x = Conv1D(512,3,activation = 'relu',padding = 'same')(x)
    x = Conv1D(512,3,activation = 'relu',padding = 'same')(x)
    x = Conv1D(512,3,activation = 'relu',padding = 'same')(x)
    x = MaxPooling1D(2,strides = 2,padding='valid')(x)
    
    x=eca_block(x)
    
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(32, activation='relu')(x)
    #out = Dense(16, activation='relu')(x)

    return Model(Ini,out)

#--2s---test
def Base_cnn_2(input_shape):
    Ini=Input(input_shape)
        
        
    # 第二个卷积部分
    x = Conv1D(64,5,activation = 'relu',padding = 'same')(Ini)
    x = MaxPooling1D(4,strides = 4,padding='valid')(x)
    
    # 第二个卷积部分
    x = Conv1D(64,5,activation = 'relu',padding = 'same')(x)
    x = MaxPooling1D(4,strides = 4,padding='valid')(x)
    
    
    x = Conv1D(128,5,activation = 'relu',padding = 'same')(x)
    x = MaxPooling1D(4,strides = 4,padding='valid')(x)
    
    x = Conv1D(256,5,activation = 'relu',padding = 'same')(x)
    x = MaxPooling1D(4,strides = 4,padding='valid')(x)
    
    x = Conv1D(512,5,activation = 'relu',padding = 'same')(x)
    x = Conv1D(512,5,activation = 'relu',padding = 'same')(x)
    x = MaxPooling1D(4,strides = 4,padding='valid')(x)
    
   
    
    x=eca_block(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(32, activation='relu')(x)
    #out = Dense(16, activation='relu')(x)

    return Model(Ini,out)

#--2s---test
def Base_cnn_3(input_shape):
    Ini=Input(input_shape)
        
        
    # 第二个卷积部分
    x = Conv1D(64,7,activation = 'relu',padding = 'same')(Ini)
    x = MaxPooling1D(6,strides = 6,padding='valid')(x)
    
    # 第二个卷积部分
    x = Conv1D(128,7,activation = 'relu',padding = 'same')(x)
    x = MaxPooling1D(6,strides = 6,padding='valid')(x)
    
    
    x = Conv1D(256,7,activation = 'relu',padding = 'same')(x)
    x = MaxPooling1D(6,strides = 6,padding='valid')(x)
    
    x = Conv1D(512,7,activation = 'relu',padding = 'same')(x)
    x = MaxPooling1D(6,strides = 6,padding='valid')(x)
    
    x=eca_block(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(32, activation='relu')(x)
    #out = Dense(16, activation='relu')(x)

    return Model(Ini,out)
        

def comb_cnn(input_shape):
    cnn_mode_1=Base_cnn_1(input_shape)
    cnn_mode_2=Base_cnn_2(input_shape)
    cnn_mode_3=Base_cnn_3(input_shape)
    Ini=Input(input_shape)
    x1=cnn_mode_1(Ini)
    x2=cnn_mode_2(Ini)
    x3=cnn_mode_3(Ini)
    x4=tf.concat([x1,x2,x3],axis=-1)
    out = Dense(32, activation='relu')(x4)
    return Model(Ini,out)
    


def CNN_LSTM(input_shape):
    #新建主干1Dcnn，该模型输入格式为(None,32000,1)
    Bc=Base_cnn([1600,1])
  
    Ini=Input(input_shape)
    
    #2s重叠1s
    temp=[]
    start=0
    end=1600
    segments=0
    while end<=Ini.shape[1]:
        x=Ini[:,start:end]
        start+=800
        end=start+1600
        out=Bc(x)
        temp.append(tf.expand_dims(out,[1]))
        segments+=1
    vec=temp[0]
    for i in range(1,len(temp)):
        vec=tf.concat([vec,temp[i]],axis=1)
    print(vec.shape)
    vec = LSTM(39)(vec)
    out= Dense(16, activation='relu')(vec)

    return Model(Ini,out)