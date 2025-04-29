import numpy as np
def nextpow2(n):
    '''
    求最接近数据长度的2的整数次方
    An integer equal to 2 that is closest to the length of the data
    
    Eg: 
    nextpow2(2) = 1
    nextpow2(2**10+1) = 11
    nextpow2(2**20+1) = 21
    '''
    return np.ceil(np.log2(np.abs(n))).astype('long')
