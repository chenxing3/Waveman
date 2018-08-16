import os
import pylab
import numpy as np
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt

"""
This module provides some short functions to process image
"""

def ConvJPG(filename, array, framerate, width, height, train=False):
    '''
    array convert to image
    ylabe of the image are uniform to 19200 or 192000 KHz
    '''
    if train == False:
        plt.figure(figsize=(2.56, 2.56), dpi=100)
    else:
        plt.figure(figsize=(width/100, height/100), dpi=100)        
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    plt.specgram(array, Fs=framerate, scale_by_freq=True, sides='default', 
                 noverlap=474, NFFT=512, cmap='jet')
    if framerate < 100000:
        yaxis = 19200
    else:
        yaxis = 192000
    plt.ylim(0, yaxis)
    pylab.savefig(filename, bbox_inches=None, pad_inches=0)
    pylab.close()
    plt.close()
    if train == False:
        # resize image
        img = Image.open(filename)
        new_img = img.resize((width, height), Image.BILINEAR)
        new_img.save(filename)


def read_and_decode_single_example(filename):
    '''
    decode image matrix and return it together with the total number of images
    '''
    with open(filename) as fid:
        content = fid.read()
    content = content.split('\n')
    content = content[:-1]

    res = []
    for i in range(len(content)):
        lena = misc.imread(content[i]).astype(np.int16)
        lena_norm = (lena - 128) / 128.0
        res.append(lena_norm)
    res = np.array(res)
    return len(content), res

