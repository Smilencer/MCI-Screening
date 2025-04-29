import os

from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input, Lambda, MaxPooling1D,concatenate
from tensorflow.keras.models import Model
from PIL import Image



#-------------------------#
#   创建triplet神经网络
#-------------------------#
def triplet_net(input_shape,Base_cnn):   
    
    #input_shape为 （3，297,39,1）6s
    #input_shape为 （3，197,39,1）4s
    input_audio=Input(shape=input_shape)
    input_audio_anchor = input_audio[:,0]
    input_audio_positive = input_audio[:,1]
    input_audio_negative = input_audio[:,2]
    #------------------------------------------#
    #   我们将三个个输入传入到主干特征提取网络
    #------------------------------------------#
    encoded_audio_anchor = Base_cnn(input_audio_anchor)
    encoded_audio_positive = Base_cnn(input_audio_positive)
    encoded_audio_negative = Base_cnn(input_audio_negative)

    
    #进行连接操作，进行输出。
    merged_vector = concatenate([encoded_audio_anchor, encoded_audio_positive, encoded_audio_negative], axis=-1)#将三个特征向量连接
    return Model(input_audio, merged_vector)





    
def triplet_loss(y_true, y_pred, alpha = 1.8):
    #其中y_true,y_pred都是张量。，单个数是0维张量
    #print(y_true)#这是label,实际上是用不着label的，这个loss是单纯的依靠 anchor与 positive，negative之间的距离决定的。
    #print(y_pred)#这是预测值
    #print(y_pred.shape.as_list())
    total_lenght = y_pred.shape.as_list()[-1]
    #[None, 12]为y_pred.shape.as_list(), 因此total_lenght==12
    

    anchor = y_pred[:,0:int(total_lenght*1/3)]#当前批次中，每个样本的anchor


    positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]#当前批次中，每个样本的positive

    negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]#当前批次中，每个样本的negative

    # distance between the anchor and the positive

    pos_dist = K.sum(K.square(anchor-positive),axis=1)#l2范数的平方,这是计算的一批的loss
    #print(pos_dist)

    # distance between the anchor and the negative

    neg_dist = K.sum(K.square(anchor-negative),axis=1)
    
    '''
    neg_Min=tf.reduce_min(neg_dist, axis=0)
    
    Mask=phe*neg_Min-neg_dist
    Mask = Mask > 0#如果有neg距离 大于一定的阈值了，那么就会对其做求强制性loss
    
    x = tf.constant(0, dtype=tf.float32)
    neg_dist=tf.where(Mask, neg_dist, x)
    '''
    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
    
    return loss


def triplet_loss_test(y_true, y_pred, alpha = 2):
    #其中y_true,y_pred都是张量。，单个数是0维张量
    #print(y_true)#这是label,实际上是用不着label的，这个loss是单纯的依靠 anchor与 positive，negative之间的距离决定的。
    #print(y_pred)#这是预测值
    #print(y_pred.shape.as_list())
    total_lenght = y_pred.shape.as_list()[-1]
    #[None, 12]为y_pred.shape.as_list(), 因此total_lenght==12
    

    anchor = y_pred[:,0:int(total_lenght*1/3)]#当前批次中，每个样本的anchor


    positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]#当前批次中，每个样本的positive

    negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]#当前批次中，每个样本的negative
    
    
    # distance between the anchor and the positive

    pos_dist = K.sum(K.square(anchor-positive),axis=1)#l2范数的平方,这是计算的一批的loss
    #print(pos_dist)

    # distance between the anchor and the negative

    neg_dist = K.sum(K.square(anchor-negative),axis=1)
    
    Min=tf.reduce_min(neg_dis, axis=0)#求得最小的neg_dist

    # compute loss

    
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
    
    Loss=[]
    Min=tf.reduce_min(loss, axis=0)
    
    print("***********")
    for i in range(16):
        Loss.append(Min)
    print("********")
    
      
    return Loss