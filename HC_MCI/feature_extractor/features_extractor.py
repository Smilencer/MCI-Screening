import scipy.io.wavfile as wav
import numpy as np
from feature_extractor.webrtc_vad_filter import WebrtcVADFilter
from feature_extractor.python_speech_features import delta, rasta
# from feature_extractor.mfcc_feature_extractor import MfccFeatureExtractor
from feature_extractor.fir_filter import FIRFilter
import librosa
import os


import scipy.io.wavfile as wav
import numpy as np
from feature_extractor.webrtc_vad_filter import WebrtcVADFilter
from feature_extractor.python_speech_features import delta, rasta
# from feature_extractor.mfcc_feature_extractor import MfccFeatureExtractor
from feature_extractor.fir_filter import FIRFilter
import librosa
import os


class extractor:
    BASE_MODE = "mfcc-base"
    DELTA_MODE = "mfcc-delta"
    DELTA_DELTA_MODE = "mfcc-delta-delta"
    RASTA_MODE = "raste"
    PLP_MODE = "plp"
    RASTA_PLP_MODE = "raste-plp"
    MFCC_RASTA_MODE = "mfcc-raste"
    
    MODES = [BASE_MODE, DELTA_MODE, DELTA_DELTA_MODE]
    def __init__(self,extract_mode="mfcc-delta-delta"):
        self.extract_mode=extract_mode
    def extract(self, signal, rate):
        #if self.extract_mode == self.RASTA_MODE || self.extract_mode == self.RASTA_PLP_MODE:
        #rasta_feat = np.transpose(rasta.rastaplp(signal, fs=rate, modelorder=12))
        
        #if self.extract_mode == self.DELTA_MODE || selfextract_mode == self.DELTA_MODE || self.extract_mode == self.BASE_MODE:
        #mfcc_feat = np.transpose(rasta.melfcc(signal, fs=rate))
        #plp_feat = np.transpose(rasta.melfcc(signal, fs=rate, fbtype='bark'))
        if self.extract_mode == self.BASE_MODE:
            mfcc_feat = np.transpose(rasta.melfcc(signal, fs=rate))
            self.feat=mfcc_feat
        if self.extract_mode == self.DELTA_MODE:
            mfcc_feat = np.transpose(rasta.melfcc(signal, fs=rate))
            mfcc_feat = np.concatenate((mfcc_feat, delta(mfcc_feat, 2)), axis=1)
            self.feat=mfcc_feat
        if self.extract_mode == self.DELTA_DELTA_MODE:
            mfcc_feat = np.transpose(rasta.melfcc(signal, fs=rate))
            delta_feat = delta(mfcc_feat, 2)
            mfcc_feat = np.concatenate((mfcc_feat, delta_feat, delta(delta_feat, 2)), axis=1) 
            self.feat=mfcc_feat
        if self.extract_mode == self.PLP_MODE:
            plp_feat = np.transpose(rasta.melfcc(signal, fs=rate, fbtype='bark'))
            self.feat=plp_feat
        if self.extract_mode == self.RASTA_PLP_MODE:
            rasta_feat = np.transpose(rasta.rastaplp(signal, fs=rate, modelorder=12))
            plp_feat = np.transpose(rasta.melfcc(signal, fs=rate, fbtype='bark'))
            self.feat=np.concatenate((rasta_feat, plp_feat), axis=1)
            
        if self.extract_mode == self.MFCC_RASTA_MODE:
            mfcc_feat = np.transpose(rasta.melfcc(signal, fs=rate))
            delta_feat = delta(mfcc_feat, 2)
            rasta_feat = np.transpose(rasta.rastaplp(signal, fs=rate, modelorder=12))
            plp_feat = np.transpose(rasta.melfcc(signal, fs=rate, fbtype='bark'))
            self.feat=np.concatenate((rasta_feat, mfcc_feat, plp_feat,delta_feat,delta(delta_feat, 2)), axis=1)
        return self.feat
    
class Data_features_extractor:
    
    BASE_MODE = "mfcc-base"
    DELTA_MODE = "mfcc-delta"
    DELTA_DELTA_MODE = "mfcc-delta-delta"
    RASTA_MODE = "raste"
    PLP_MODE = "plp"
    RASTA_PLP_MODE = "raste-plp"
    MFCC_RASTA_MODE = "mfcc-raste"
    
    MODES = [BASE_MODE, DELTA_MODE, DELTA_DELTA_MODE]
    
    
    def __init__(self, trainRtest,extract_mode="mfcc-delta-delta",All=True):
        self.extract_mode=extract_mode    
        self.datas=[]
        self.win_length = 0.03
        self.cur_path=os.path.join(os.getcwd(),'split_data',trainRtest)

        #获取其中类别，AD,HC,MCI
        self.Classes=os.listdir(self.cur_path)
        count=-1
        
        for Class in self.Classes:
            count+=1
            self.datas.append([])
            files=os.listdir(os.path.join(self.cur_path,Class))
            for file in files:
                # use webrtc VAD for filtering:
                rate, signal = wav.read(os.path.join(self.cur_path,Class,file))
                original_len = len(signal)
                print(Class,"File %s audio file result: %d/%d" % (file, len(signal), original_len))
                if All==True:
                    split_array=self.split_accu(signal)
                else:
                    split_array=self.split_accu_all(signal)
                   
                mark=1
                for split in split_array:
                    if mark==1:
                        print(self.extract(split,rate).shape)
                        mark=0
                    self.datas[count].append(self.extract(split,rate))
                print('Extracted file: {}'.format(file))


    # #剔除低能量音频段
    def get_E(self,arr):
         #将arr绝对值化
        Abs=np.maximum(arr,-arr)
        #求得总的平均值
        return np.mean(Abs)
    
    #，同时只选取有效音频段
    def split_accu(self,arr):
    
        arr_E=self.get_E(arr)
        new_arr=[]
        start=0
        end=32000
        while True:
            mark=0
            if len(arr)-start<32000:
                zeros = [0] * (len(arr)-start)
                arr=np.concatenate((arr,zeros), axis=0) 
                mark=1
            if self.get_E(arr[start:end])>arr_E*0.3:
                new_arr.append(arr[start:end])
            if mark==1:
                break
            start+=16000
            end=start+32000
            if end>len(arr):
                break
        new_arr=np.array(new_arr)
        return new_arr
    
    #以选取全部音频
    def split_accu_all(self,arr):
    
        arr_E=self.get_E(arr)
        new_arr=[]
        start=0
        end=32000
        while True:
            mark=0
            if len(arr)-start<32000:
                zeros = [0] * (len(arr)-start)
                arr=np.concatenate((arr,zeros), axis=0) 
                mark=1
            
            new_arr.append(arr[start:end])
            if mark==1:
                break
            start+=16000
            end=start+32000
            if end>len(arr):
                break
        new_arr=np.array(new_arr)
        return new_arr

    
    #读取音频数据，将其转换为对应特征谱
    def extract(self, signal, rate):
        #if self.extract_mode == self.RASTA_MODE || self.extract_mode == self.RASTA_PLP_MODE:
        #rasta_feat = np.transpose(rasta.rastaplp(signal, fs=rate, modelorder=12))
        
        #if self.extract_mode == self.DELTA_MODE || selfextract_mode == self.DELTA_MODE || self.extract_mode == self.BASE_MODE:
        #mfcc_feat = np.transpose(rasta.melfcc(signal, fs=rate))
        #plp_feat = np.transpose(rasta.melfcc(signal, fs=rate, fbtype='bark'))
        if self.extract_mode == self.BASE_MODE:
            mfcc_feat = np.transpose(rasta.melfcc(signal, fs=rate))
            self.feat=mfcc_feat
        if self.extract_mode == self.DELTA_MODE:
            mfcc_feat = np.transpose(rasta.melfcc(signal, fs=rate))
            mfcc_feat = np.concatenate((mfcc_feat, delta(mfcc_feat, 2)), axis=1)
            self.feat=mfcc_feat
        if self.extract_mode == self.DELTA_DELTA_MODE:
            mfcc_feat = np.transpose(rasta.melfcc(signal, fs=rate))
            delta_feat = delta(mfcc_feat, 2)
            mfcc_feat = np.concatenate((mfcc_feat, delta_feat, delta(delta_feat, 2)), axis=1) 
            self.feat=mfcc_feat
        if self.extract_mode == self.PLP_MODE:
            plp_feat = np.transpose(rasta.melfcc(signal, fs=rate, fbtype='bark'))
            self.feat=plp_feat
        if self.extract_mode == self.RASTA_PLP_MODE:
            rasta_feat = np.transpose(rasta.rastaplp(signal, fs=rate, modelorder=12))
            plp_feat = np.transpose(rasta.melfcc(signal, fs=rate, fbtype='bark'))
            self.feat=np.concatenate((rasta_feat, plp_feat), axis=1) 
        if self.extract_mode == self.MFCC_RASTA_MODE:
            mfcc_feat = np.transpose(rasta.melfcc(signal, fs=rate))
            delta_feat = delta(mfcc_feat, 2)
            rasta_feat = np.transpose(rasta.rastaplp(signal, fs=rate, modelorder=12))
            plp_feat = np.transpose(rasta.melfcc(signal, fs=rate, fbtype='bark'))
            self.feat=np.concatenate((rasta_feat, mfcc_feat, plp_feat,delta_feat,delta(delta_feat, 2)), axis=1)
        return self.feat
        #feature_vector = np.concatenate((rasta_feat, mfcc_feat, plp_feat), axis=1)  # mfcc + rasta + plp
        # feature_vector = np.concatenate((mfcc_feat, plp_feat), axis=1)            # mfcc + plp
        # feature_vector = rasta_feat                                               # rasta
        # feature_vector = plp_feat                                                 # plp
        # feature_vector = mfcc_feat                                                # mfcc

#         if self.feature_vectors.size == 0:
#             self.feature_vectors = np.copy(feature_vector)
#         else:
#             self.feature_vectors = np.concatenate((self.feature_vectors, feature_vector))

        # return feature_vector
  

    def get_data(self):
        return np.array(self.datas)
        