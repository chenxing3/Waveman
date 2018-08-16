
'''
Covert audio file to image

'''

import os
import sys
import time
import shutil
import string
import logging
import random
import librosa
import argparse
from utils import img_utils, ops


def parse_args(path):
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser('Convert_Image.py')
    parser.add_argument('--AudioFile', dest='AudioFile', type=str,
                        help='(require)Please enter audio file for prediction')
    parser.add_argument('--ID', dest='ID', type=str,
                        default = ''.join(random.sample(string.ascii_letters 
                            + string.digits, 8)), # default is random ID
                        help='Give specific ID to audio file')
    parser.add_argument('--ImgDir', dest='ImgDir', type=str,
                        default = path+'/image',
                        help='image and other temp files')
    parser.add_argument('--ImageWidth', dest='ImageWidth', type=int,
                        default= 256,
                        help='(optimal) image width, default is 64')
    parser.add_argument('--ImageHeight', dest='ImageHeight', type=int,
                        default= 256,
                        help='(optimal) image height, default is 64')
    parser.add_argument('--SegmentLength', dest='SegmentLength', type=int,
                        default= 10368, 
                        help='(optimal) length of segment, default is 10368')
    args = parser.parse_args()
    
    # print help information
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    if args.ImageWidth < 1:
        print('\n-------------->')
        print('Error! image width must be an integer that >= 1')
        sys.exit(1)

    if args.ImageHeight < 1:
        print('\n-------------->')
        print('Error! image height must be an integer that >= 1')
        sys.exit(1)

    if args.SegmentLength < 1:
        print('\n-------------->')
        print('Error! Audio unit length must be an integer that >= 1')
        sys.exit(1)

    return args

def main():
    time_start = time.time()
    # get absolute path of Waveman and parse input arguments
    Path, _ = os.path.split(os.path.abspath(sys.argv[0]))
    args = parse_args(Path.replace('\\','/'))
    
    try:
        signal, SampleRate = librosa.load(args.AudioFile, sr=None) # read audio file
    except:
        print('WARNING: error in wav file')
        sys.exit(1)

    if args.ImgDir != (Path+'/image'): # if not define the folder to store image
        Dir = os.path.dirname(args.AudioFile)+'/'+args.ID
    else: # if defined
        Dir = args.ImgDir+'/'+args.ID

    ops.ChkDir(Dir)

    # log file 
    LogFile = Dir+'/log.txt'
    if os.path.isfile(LogFile):
        os.remove(LogFile)

    log = ops.Logs(logging.getLogger(), LogFile, logging.NOTSET)
    log.info('\nConvert file {} to images.\n'.format(args.AudioFile))
    log.info(' '.join(sys.argv) + '\n')
    log.info('The images in the folder: {}\n'.format(Dir))

    log = ops.Logs(log, LogFile, logging.INFO)
    log.info('Creat folders for assign images!')
    # prepare folder for manual assigning images
    ops.ChkDir(Dir+'/nosignal') # store no signal episodes
    ops.ChkDir(Dir+'/weak') # store weak signal episodes
    ops.ChkDir(Dir+'/strong') # store weak strong episodes
    ops.ChkDir(Dir+'/other') # store weak echo or other episodes
    ops.ChkDir(Dir+'/raw') # backup audio files

    # Backup the audio file
    BackupFile = Dir+'/raw/'+os.path.basename(args.AudioFile)
    if not os.path.isfile(BackupFile):
        shutil.copyfile(args.AudioFile, BackupFile)

    # step 1. split array
    log.info('Split array!')
    signal_matrix, indexes = ops.GetEpisodes(signal, args.SegmentLength) 

    # step 2. convert signal array to images
    log.info('Convert signal array to {} images'.format(len(indexes)))
    image_files = []
    for i in range(len(indexes)):
        TargetImg = Dir+'/'+args.ID+'-'+str(i).zfill(5)+'.jpg' # conver array to image
        img_utils.ConvJPG(TargetImg, signal_matrix[i], SampleRate, 
            args.ImageWidth, args.ImageHeight, train=True) 
        image_files.append(TargetImg)

        if i % (int(len(indexes)/10)+1) == 0:
            log.info('Genereted {} images'.format(str(i)))

    log = ops.Logs(logging.getLogger(), LogFile, logging.DEBUG)
    log.info("\nTotal cost " + str(time.time() - time_start))
    log.debug('\nPlease contract us if you find bugs! \nE-mail: chenxing@mail.xtbg.ac.cn ')

if __name__ == '__main__':
    main()

''' Copyright 2018 '''
